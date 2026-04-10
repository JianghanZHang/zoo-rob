"""RAC (Resolvent Actor-Critic) — CleanRL-style single-file implementation.

Compatible with zoo-rob benchmark. Follows CleanRL conventions:
  - Single file, self-contained
  - tyro CLI, TensorBoard/W&B logging
  - gymnasium.vector environments

Algorithm (matches paper Algorithm 1 exactly):
  1. Collect rollout under pi_theta
  2. Compute done-masked n-step returns G^(n)_t
  3. Omega update (K_omega steps): minimize ||F^(omega) - V||^2 w.r.t. psi
  4. Critic update (K_v steps): minimize ||F^(omega) - V_phi||^2 w.r.t. phi
  5. Actor update (K_a steps): vanilla PG with advantage A = F^(omega) - V_phi

Clean stack: no PPO clipping, no target networks.
Stability from: obs/reward normalization + DAgger buffer + learned omega.

Usage:
  python rac_cleanrl.py --env_id HalfCheetah-v4 --total_timesteps 1000000
  python rac_cleanrl.py --env_id HalfCheetah-v4 --learn_omega False  # fixed-omega ablation
"""

import os
import random
import time
from dataclasses import dataclass

import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
from torch.distributions.normal import Normal
from torch.utils.tensorboard import SummaryWriter


@dataclass
class Args:
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = 1
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "td-omega"
    wandb_entity: str = None
    capture_video: bool = False

    # Environment
    env_id: str = "HalfCheetah-v4"
    total_timesteps: int = 1000000
    num_envs: int = 1
    num_steps: int = 2048

    # Learning rates (three-timescale: omega > critic > actor)
    lr_actor: float = 3e-4
    lr_critic: float = 3e-4
    lr_omega: float = 1e-3

    # TD-omega
    N: int = 8
    gamma: float = 0.99
    learn_omega: bool = False
    fixed_lambda: float = 0.95  # used when learn_omega=False

    # Three-timescale update counts
    omega_epochs: int = 3
    critic_epochs: int = 2
    actor_epochs: int = 1

    # Normalization (preprocessing, not ad-hoc tricks)
    norm_obs: bool = True
    norm_reward: bool = True
    obs_clip: float = 10.0
    reward_clip: float = 10.0

    # Entropy
    ent_coef: float = 0.0

    # Gradient norm (set to 0 to disable; only for obs/reward-normalized inputs)
    max_grad_norm: float = 0.5


# ─── Network ──────────────────────────────────────────────

def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self, envs, N=8):
        super().__init__()
        obs_dim = np.array(envs.single_observation_space.shape).prod()
        act_dim = np.prod(envs.single_action_space.shape)

        self.critic = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 1), std=1.0),
        )
        self.actor_mean = nn.Sequential(
            layer_init(nn.Linear(obs_dim, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, 256)), nn.Tanh(),
            layer_init(nn.Linear(256, act_dim), std=0.01),
        )
        self.actor_logstd = nn.Parameter(torch.zeros(1, act_dim))
        self.omega_logits = nn.Parameter(torch.zeros(N))

    def get_value(self, x):
        return self.critic(x)

    def get_action_and_value(self, x, action=None):
        action_mean = self.actor_mean(x)
        action_std = self.actor_logstd.expand_as(action_mean).exp()
        probs = Normal(action_mean, action_std)
        if action is None:
            action = probs.sample()
        return action, probs.log_prob(action).sum(1), probs.entropy().sum(1), self.critic(x)

    @property
    def omega(self):
        return F.softmax(self.omega_logits, dim=0)


# ─── Normalization (preprocessing, not ad-hoc) ───────────

class RunningMeanStd:
    def __init__(self, shape):
        self.mean = np.zeros(shape, dtype=np.float64)
        self.var = np.ones(shape, dtype=np.float64)
        self.count = 1e-4

    def update(self, x):
        batch_mean = np.mean(x, axis=0)
        batch_var = np.var(x, axis=0)
        batch_count = x.shape[0]
        delta = batch_mean - self.mean
        tot = self.count + batch_count
        self.mean = self.mean + delta * batch_count / tot
        m_a = self.var * self.count
        m_b = batch_var * batch_count
        self.var = (m_a + m_b + np.square(delta) * self.count * batch_count / tot) / tot
        self.count = tot


# ─── N-step returns with done masking ─────────────────────

def compute_nstep_returns(rewards, values, dones, gamma, n):
    """Vectorized n-step returns. Matches paper Algorithm 1 lines 6-10."""
    T = len(rewards)
    device = rewards.device
    t_idx = torch.arange(T, device=device)
    offsets = torch.arange(n, device=device)
    idx = t_idx[:, None] + offsets[None, :]
    in_bounds = idx < T
    safe_idx = idx.clamp(max=T - 1)

    # Episode boundary masking
    r_vals = rewards[safe_idx] * in_bounds.float()
    done_vals = dones[safe_idx]
    not_done = torch.ones_like(done_vals)
    for i in range(1, n):
        not_done[:, i] = not_done[:, i-1] * (1 - done_vals[:, i-1])
    r_vals = r_vals * not_done * in_bounds.float()

    disc = gamma ** offsets.float()
    disc_sum = (r_vals * disc[None, :]).sum(dim=1)

    # Bootstrap: zero at terminal, V(s_{t+n}) otherwise
    hit_terminal = (not_done * done_vals * in_bounds.float()).sum(dim=1) > 0
    cum_nodone = (not_done * (1 - done_vals) * in_bounds.float()).sum(dim=1)
    eff_n = cum_nodone.long()
    boot_idx = (t_idx + eff_n).clamp(max=T)
    bootstrap = (gamma ** eff_n.float()) * values[boot_idx]
    bootstrap = bootstrap * (~hit_terminal).float()

    return disc_sum + bootstrap


# ─── Main ─────────────────────────────────────────────────

if __name__ == "__main__":
    args = tyro.cli(Args)
    run_name = f"{args.env_id}__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb
        wandb.init(project=args.wandb_project_name, entity=args.wandb_entity,
                   config=vars(args), name=run_name, save_code=True)
    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text("hyperparameters",
                    "|param|value|\n|-|-|\n" + "\n".join(
                        f"|{k}|{v}|" for k, v in vars(args).items()))

    # Seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic
    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # Environment (with RecordEpisodeStatistics for proper return logging)
    envs = gym.vector.SyncVectorEnv([
        lambda: gym.wrappers.RecordEpisodeStatistics(gym.make(args.env_id))
        for _ in range(args.num_envs)])
    assert isinstance(envs.single_action_space, gym.spaces.Box)

    agent = Agent(envs, N=args.N).to(device)

    # If fixed omega, set to geometric and freeze
    if not args.learn_omega:
        with torch.no_grad():
            lam = args.fixed_lambda
            w = np.array([(1-lam)*lam**(n-1) for n in range(1, args.N+1)])
            w /= w.sum()
            agent.omega_logits.data = torch.log(torch.tensor(w, device=device) + 1e-10)

    # Three separate optimizers (three-timescale)
    actor_optimizer = optim.Adam(
        list(agent.actor_mean.parameters()) + [agent.actor_logstd],
        lr=args.lr_actor)
    critic_optimizer = optim.Adam(agent.critic.parameters(), lr=args.lr_critic)
    omega_optimizer = optim.Adam(
        [agent.omega_logits], lr=args.lr_omega) if args.learn_omega else None

    # Normalization
    obs_rms = RunningMeanStd(envs.single_observation_space.shape) if args.norm_obs else None

    def normalize_obs(obs_np):
        if obs_rms is not None:
            obs_rms.update(obs_np)
            return np.clip((obs_np - obs_rms.mean) / (np.sqrt(obs_rms.var) + 1e-8),
                           -args.obs_clip, args.obs_clip).astype(np.float32)
        return obs_np.astype(np.float32)

    class RewardNormalizer:
        def __init__(self, num_envs, gamma, clip):
            self.ret = np.zeros(num_envs)
            self.rms = RunningMeanStd(())
            self.gamma = gamma
            self.clip = clip
        def normalize(self, rew_np):
            self.ret = self.ret * self.gamma + rew_np
            self.rms.update(self.ret.reshape(-1, 1))
            return np.clip(rew_np / (np.sqrt(self.rms.var) + 1e-8),
                           -self.clip, self.clip).astype(np.float32)
        def reset(self, idx):
            self.ret[idx] = 0.0

    rew_norm = RewardNormalizer(args.num_envs, args.gamma, args.reward_clip) \
        if args.norm_reward else None

    # Storage
    obs = torch.zeros((args.num_steps, args.num_envs) +
                      envs.single_observation_space.shape).to(device)
    actions = torch.zeros((args.num_steps, args.num_envs) +
                          envs.single_action_space.shape).to(device)
    rewards = torch.zeros((args.num_steps, args.num_envs)).to(device)
    dones = torch.zeros((args.num_steps, args.num_envs)).to(device)
    values = torch.zeros((args.num_steps, args.num_envs)).to(device)

    # Start
    global_step = 0
    start_time = time.time()
    next_obs, _ = envs.reset(seed=args.seed)
    next_obs = normalize_obs(next_obs)
    next_obs = torch.Tensor(next_obs).to(device)
    next_done = torch.zeros(args.num_envs).to(device)
    num_updates = args.total_timesteps // (args.num_steps * args.num_envs)

    for update in range(1, num_updates + 1):
        # ── Collect rollout (Paper Algorithm 1, line 2) ──
        for step in range(args.num_steps):
            global_step += args.num_envs
            obs[step] = next_obs
            dones[step] = next_done

            with torch.no_grad():
                action, logprob, _, value = agent.get_action_and_value(next_obs)
                values[step] = value.flatten()
            actions[step] = action

            next_obs_np, reward_np, terminations, truncations, infos = envs.step(
                action.cpu().numpy())
            next_done_np = np.logical_or(terminations, truncations)

            # Reset running return at episode boundary
            if rew_norm is not None:
                for i, d in enumerate(next_done_np):
                    if d:
                        rew_norm.reset(i)

            reward_np = rew_norm.normalize(reward_np) if rew_norm else reward_np.astype(np.float32)
            rewards[step] = torch.tensor(reward_np).to(device).view(-1)
            next_obs_np = normalize_obs(next_obs_np)
            next_obs = torch.Tensor(next_obs_np).to(device)
            next_done = torch.Tensor(next_done_np.astype(np.float32)).to(device)

            # Log episode returns
            if "final_info" in infos:
                for info in infos["final_info"]:
                    if info is not None and "episode" in info:
                        writer.add_scalar("charts/episodic_return",
                                          info["episode"]["r"], global_step)
                        writer.add_scalar("charts/episodic_length",
                                          info["episode"]["l"], global_step)
                        if update % 5 == 0:
                            print(f"  step={global_step:>8d}  "
                                  f"ep_return={info['episode']['r']:.1f}  "
                                  f"ep_len={info['episode']['l']}",
                                  flush=True)

        # Bootstrap value (Paper Algorithm 1, line 3)
        with torch.no_grad():
            next_value = agent.get_value(next_obs).reshape(1, -1)

        b_obs = obs.reshape(-1, *envs.single_observation_space.shape)
        b_actions = actions.reshape(-1, *envs.single_action_space.shape)
        b_rewards = rewards.reshape(-1)
        b_dones = dones.reshape(-1)
        b_values = torch.cat([values.reshape(-1), next_value.reshape(-1)])

        # ── Omega update: shape the resolvent (Paper Algorithm 1, lines 12-16) ──
        if omega_optimizer is not None:
            for _ in range(args.omega_epochs):
                omega = agent.omega
                G_list = [compute_nstep_returns(b_rewards, b_values, b_dones,
                                                args.gamma, n)
                          for n in range(1, args.N + 1)]
                G_stack = torch.stack(G_list)
                F_omega = torch.einsum('n,nt->t', omega, G_stack)
                omega_loss = ((F_omega - b_values[:-1].detach()) ** 2).mean()

                omega_optimizer.zero_grad()
                omega_loss.backward()
                omega_optimizer.step()

        # ── Critic update: fit V to omega-weighted target (Paper lines 17-19) ──
        with torch.no_grad():
            omega = agent.omega
            G_list = [compute_nstep_returns(b_rewards, b_values, b_dones,
                                            args.gamma, n)
                      for n in range(1, args.N + 1)]
            targets = torch.einsum('n,nt->t', omega, torch.stack(G_list))

        for _ in range(args.critic_epochs):
            v = agent.get_value(b_obs).squeeze(-1)
            critic_loss = F.mse_loss(v, targets)

            critic_optimizer.zero_grad()
            critic_loss.backward()
            if args.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(agent.critic.parameters(),
                                         args.max_grad_norm)
            critic_optimizer.step()

        # ── Actor update: vanilla PG (Paper lines 20-23) ──
        with torch.no_grad():
            v_new = agent.get_value(b_obs).squeeze(-1)
            advantages = targets - v_new
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

        for _ in range(args.actor_epochs):
            _, newlogprob, entropy, _ = agent.get_action_and_value(
                b_obs, b_actions)
            actor_loss = -(newlogprob * advantages).mean()
            actor_loss = actor_loss - args.ent_coef * entropy.mean()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            if args.max_grad_norm > 0:
                nn.utils.clip_grad_norm_(
                    list(agent.actor_mean.parameters()) + [agent.actor_logstd],
                    args.max_grad_norm)
            actor_optimizer.step()

        # ── Logging ──
        omega_np = agent.omega.detach().cpu().numpy()
        writer.add_scalar("losses/value_loss", critic_loss.item(), global_step)
        writer.add_scalar("losses/actor_loss", actor_loss.item(), global_step)
        if omega_optimizer is not None:
            writer.add_scalar("losses/omega_loss", omega_loss.item(), global_step)
        for i in range(min(4, args.N)):
            writer.add_scalar(f"omega/w{i+1}", omega_np[i], global_step)
        writer.add_scalar("charts/SPS",
                          int(global_step / (time.time() - start_time)),
                          global_step)

    envs.close()
    writer.close()
