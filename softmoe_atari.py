import os
import random
import time
from dataclasses import dataclass
from collections import deque

import ale_py
import gymnasium as gym
gym.register_envs(ale_py)

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import tyro
from stable_baselines3.common.atari_wrappers import (
    ClipRewardEnv,
    EpisodicLifeEnv,
    FireResetEnv,
    MaxAndSkipEnv,
    NoopResetEnv,
)
from stable_baselines3.common.buffers import ReplayBuffer
from torch.utils.tensorboard import SummaryWriter


# -------------------- Args --------------------
@dataclass
class Args:
    # run / logging
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    seed: int = random.randint(0,1000)
    torch_deterministic: bool = True
    cuda: bool = True
    track: bool = False
    wandb_project_name: str = "cleanRL"
    wandb_entity: str = None
    capture_video: bool = False
    save_model: bool = False
    upload_model: bool = False
    hf_entity: str = ""
    env_id: str = "SpaceInvadersNoFrameskip-v4"
    num_envs: int = 1
    total_timesteps: int = 2_000_000
    learning_rate: float = 1e-4
    gamma: float = 0.99
    buffer_size: int = 1_000_000
    batch_size: int = 256
    train_frequency: int = 4           
    updates_per_step: int = 1          
    learning_starts: int = 80_000
    start_e: float = 1.0
    end_e: float = 0.01
    exploration_fraction: float = 0.10
    tau: float = 1.0                   
    target_network_frequency: int = 1_000
    double_dqn: bool = False           
    model_type: str = "moe"     

    n_experts: int = 5                
    moe_temp: float = 1.0
    moe_gate_hidden: int = 0         
    moe_entropy_coef: float = 5e-4    
    moe_balance_coef: float = 5e-4   
    wide_multiplier: int = 5    
    eval_interval: int = 100_000
    eval_episodes: int = 10
    eval_epsilon: float = 0.05
    target_return: float = 10000.0    


def make_env(env_id, seed, idx, capture_video, run_name):
    def thunk():
        if capture_video and idx == 0:
            env = gym.make(env_id, render_mode="rgb_array")
            env = gym.wrappers.RecordVideo(env, f"videos/{run_name}")
        else:
            env = gym.make(env_id)
        env = gym.wrappers.RecordEpisodeStatistics(env)
        env = NoopResetEnv(env, noop_max=30)
        env = MaxAndSkipEnv(env, skip=4)
        env = EpisodicLifeEnv(env)
        if "FIRE" in env.unwrapped.get_action_meanings():
            env = FireResetEnv(env)
        env = ClipRewardEnv(env)
        env = gym.wrappers.ResizeObservation(env, (84, 84))
        env = gym.wrappers.GrayScaleObservation(env)
        env = gym.wrappers.FrameStack(env, 4)
        env.action_space.seed(seed)
        return env
    return thunk


class DQNCNNEncoder(nn.Module):
    def __init__(self, in_ch: int = 4, out_dim: int = 512):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, 32, 8, stride=4), nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),   nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),   nn.ReLU(),
            nn.Flatten(),
        )
        self.fc = nn.Sequential(
            nn.Linear(3136, out_dim), nn.ReLU(),
        )

    def forward(self, x):
        if x.ndim == 4 and x.shape[1] not in (1,4):
            x = x.permute(0,3,1,2)
        x = x / 255.0
        return self.fc(self.conv(x))


class SoftMoELayer(nn.Module):

    def __init__(self, feat_dim=512, n_experts=4, moe_temp=1.0, gate_hidden=0):
        super().__init__()
        self.n_experts = n_experts
        self.moe_temp = moe_temp

        self.experts = nn.ModuleList([nn.Linear(feat_dim, feat_dim) for _ in range(n_experts)])
        if gate_hidden and gate_hidden > 0:
            self.gate = nn.Sequential(
                nn.Linear(feat_dim, gate_hidden), nn.ReLU(),
                nn.Linear(gate_hidden, n_experts),
            )
        else:
            self.gate = nn.Linear(feat_dim, n_experts)

    def forward(self, feats):
        logits = self.gate(feats) / max(1e-6, self.moe_temp)
        gate_p = torch.softmax(logits, dim=-1)  # (B,K)
        expert_feats = torch.stack([exp(feats) for exp in self.experts], dim=-1)  # (B,D,K)
        z = (expert_feats * gate_p.unsqueeze(1)).sum(dim=-1)  # (B,D)
        return z, gate_p


class SoftMoEDQN(nn.Module):

    def __init__(self, env, feat_dim: int = 512, n_experts: int = 4, moe_temp: float = 1.0, gate_hidden: int = 0):
        super().__init__()
        self.n_actions = env.single_action_space.n
        self.encoder = DQNCNNEncoder(in_ch=4, out_dim=feat_dim)
        self.moe = SoftMoELayer(feat_dim=feat_dim, n_experts=n_experts, moe_temp=moe_temp, gate_hidden=gate_hidden)
        self.head = nn.Linear(feat_dim, self.n_actions)

    def forward(self, obs, return_gate=False):
        feats = self.encoder(obs)
        moe_feats, gate_p = self.moe(feats)
        q = self.head(moe_feats)
        return (q, gate_p) if return_gate else q


class WidePenultDQN(nn.Module):

    def __init__(self, env, feat_dim: int = 512, width: int = 4):
        super().__init__()
        self.n_actions = env.single_action_space.n
        self.encoder = DQNCNNEncoder(in_ch=4, out_dim=feat_dim)
        self.penult = nn.Sequential(
            nn.Linear(feat_dim, feat_dim * width),
            nn.ReLU(),
        )
        self.head = nn.Linear(feat_dim * width, self.n_actions)

    def forward(self, obs):
        feats = self.encoder(obs)
        z = self.penult(feats)
        return self.head(z)


class EpisodeMonitor:
    def __init__(self, target_return=19, window_size=50, writer=None):
        self.returns = deque(maxlen=window_size)
        self.target_return = target_return
        self.window_size = window_size
        self.writer = writer

    def update(self, episode_return, global_step):
        r = float(episode_return[0]) if hasattr(episode_return, "__len__") else float(episode_return)
        self.returns.append(r)
        avg = sum(self.returns) / len(self.returns) if self.returns else 0.0
        print(f"Episode return: {r:.2f}, Running average ({len(self.returns)}/{self.window_size}): {avg:.2f}")
        if self.writer is not None:
            self.writer.add_scalar("charts/running_average_return", avg, global_step)
            self.writer.add_scalar("charts/target_return", self.target_return, global_step)

    def should_stop(self):
        if len(self.returns) < self.window_size:
            return False
        return (sum(self.returns) / self.window_size) > self.target_return


def linear_schedule(start_e: float, end_e: float, duration: int, t: int):
    slope = (end_e - start_e) / float(duration)
    return max(slope * t + start_e, end_e)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)


@torch.no_grad()
def evaluate(q_net, env_id, device, episodes, epsilon, seed=123):
    env = gym.make(env_id)
    env = gym.wrappers.RecordEpisodeStatistics(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    env = EpisodicLifeEnv(env)
    if "FIRE" in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ClipRewardEnv(env)
    env = gym.wrappers.ResizeObservation(env, (84, 84))
    env = gym.wrappers.GrayScaleObservation(env)
    env = gym.wrappers.FrameStack(env, 4)

    returns = []
    for ep in range(episodes):
        obs, _ = env.reset(seed=seed + ep)
        done = False
        ep_ret = 0.0
        while not done:
            if random.random() < epsilon:
                action = env.action_space.sample()
            else:
                obs_t = torch.as_tensor(np.expand_dims(obs, 0), device=device)
                q = q_net(obs_t)
                action = int(torch.argmax(q, dim=1).item())
            obs, r, terminated, truncated, _ = env.step(action)
            done = bool(terminated or truncated)
            ep_ret += float(r)
        returns.append(ep_ret)
    env.close()
    return float(np.mean(returns) if returns else 0.0), float(np.median(returns) if returns else 0.0)


# -------------------- Main --------------------
if __name__ == "__main__":
    import stable_baselines3 as sb3
    if sb3.__version__ < "2.0":
        raise ValueError("Requires stable_baselines3>=2.0")

    args = tyro.cli(Args)
    assert args.num_envs == 1, "vectorized envs not supported for this script"

    run_name = f"{args.env_id}__{args.model_type}__penult__{args.exp_name}__{args.seed}__{int(time.time())}"

    if args.track:
        import wandb
        wandb.init(
            project=args.wandb_project_name,
            entity=args.wandb_entity,
            sync_tensorboard=True,
            config=vars(args),
            name=run_name,
            monitor_gym=True,
            save_code=True,
        )

    writer = SummaryWriter(f"runs/{run_name}")
    writer.add_text(
        "hyperparameters",
        "|param|value|\n|-|-|\n%s" % ("\n".join([f"|{k}|{v}|" for k, v in vars(args).items()])),
    )

    monitor = EpisodeMonitor(target_return=args.target_return, window_size=50, writer=writer)

    # seeding
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.backends.cudnn.deterministic = args.torch_deterministic

    device = torch.device("cuda" if torch.cuda.is_available() and args.cuda else "cpu")

    # envs
    envs = gym.vector.SyncVectorEnv(
        [make_env(args.env_id, args.seed + i, i, args.capture_video, run_name) for i in range(args.num_envs)]
    )
    assert isinstance(envs.single_action_space, gym.spaces.Discrete), "discrete action space only"

    if args.model_type == "moe":
        q_network = SoftMoEDQN(
            envs, n_experts=args.n_experts, moe_temp=args.moe_temp, gate_hidden=args.moe_gate_hidden
        ).to(device)
        target_network = SoftMoEDQN(
            envs, n_experts=args.n_experts, moe_temp=args.moe_temp, gate_hidden=args.moe_gate_hidden
        ).to(device)
    elif args.model_type == "wide":
        q_network = WidePenultDQN(envs, width=args.wide_multiplier).to(device)
        target_network = WidePenultDQN(envs, width=args.wide_multiplier).to(device)
    else:
        raise ValueError("model_type must be one of {'moe', 'wide'}")

    target_network.load_state_dict(q_network.state_dict())
    optimizer = optim.Adam(q_network.parameters(), lr=args.learning_rate)

    print(f"[params] online : {count_params(q_network):,}")
    print(f"[params] target : {count_params(target_network):,}")

    rb = ReplayBuffer(
        args.buffer_size,
        envs.single_observation_space,
        envs.single_action_space,
        device,
        optimize_memory_usage=True,
        handle_timeout_termination=False,
    )

    obs, _ = envs.reset(seed=args.seed)
    global_step = 0
    start_time = time.time()

    if not (obs.ndim == 4 and (obs.shape[1] in (1, 4) or obs.shape[-1] in (1, 4))):
        raise RuntimeError(f"Unexpected observation shape {obs.shape}. Expected NCHW or NHWC with C in {{1,4}}.")

    while global_step < args.total_timesteps:
        epsilon = linear_schedule(args.start_e, args.end_e, int(args.exploration_fraction * args.total_timesteps), global_step)

        if random.random() < epsilon:
            actions = np.array([envs.single_action_space.sample() for _ in range(envs.num_envs)])
        else:
            q_values = q_network(torch.as_tensor(obs, device=device))
            actions = torch.argmax(q_values, dim=1).cpu().numpy()

        next_obs, rewards, terminations, truncations, infos = envs.step(actions)

        if "final_info" in infos:
            for info in infos["final_info"]:
                if info and "episode" in info:
                    ep_r = info["episode"]["r"]
                    ep_l = info["episode"]["l"]
                    print(f"global_step={global_step}, episodic_return={ep_r[0] if hasattr(ep_r,'__len__') else ep_r}")
                    writer.add_scalar("charts/episodic_return", ep_r, global_step)
                    writer.add_scalar("charts/episodic_length", ep_l, global_step)
                    monitor.update(ep_r, global_step)
                    if monitor.should_stop():
                        print(f"Solved at step {global_step}! avg(last {monitor.window_size})="
                              f"{sum(monitor.returns)/monitor.window_size:.2f}")
                        envs.close(); writer.close(); raise SystemExit(0)

        # time-limit aware final obs
        real_next_obs = next_obs.copy()
        for idx, trunc in enumerate(truncations):
            if trunc and "final_observation" in infos:
                real_next_obs[idx] = infos["final_observation"][idx]

        rb.add(obs, real_next_obs, actions, rewards, terminations, infos)
        obs = next_obs
        global_step += 1

        if (global_step % args.eval_interval == 0) or (global_step == args.total_timesteps):
            mean_r, median_r = evaluate(q_network, args.env_id, device, args.eval_episodes, args.eval_epsilon, seed=args.seed + 999)
            writer.add_scalar("eval/mean_return", mean_r, global_step)
            writer.add_scalar("eval/median_return", median_r, global_step)
            print(f"[eval] step={global_step} mean={mean_r:.2f} median={median_r:.2f}")

        if global_step > args.learning_starts and (global_step % args.train_frequency == 0):
            for _ in range(args.updates_per_step):
                data = rb.sample(args.batch_size)
                with torch.no_grad():
                    if args.double_dqn:
                        next_q_online = q_network(data.next_observations)
                        next_actions = next_q_online.argmax(dim=1, keepdim=True)          # (B,1)
                        next_q_target = target_network(data.next_observations)
                        target_next = next_q_target.gather(1, next_actions).squeeze(1)    # (B,)
                    else:
                        target_next = target_network(data.next_observations).max(dim=1)[0]
                    td_target = data.rewards.flatten() + args.gamma * target_next * (1 - data.dones.flatten().float())

                q = q_network(data.observations)
                chosen_q = q.gather(1, data.actions).squeeze(1)
                td_loss = (td_target - chosen_q).pow(2).mean()

                aux_loss = torch.tensor(0.0, device=td_loss.device)
                if args.model_type == "moe":
                    _, gate_p = q_network(data.observations, return_gate=True)  # (B,K)
                    gate_entropy = -(gate_p.clamp_min(1e-8) * gate_p.clamp_min(1e-8).log()).sum(dim=-1).mean()
                    p_bar = gate_p.mean(dim=0)
                    uniform = torch.full_like(p_bar, 1.0 / gate_p.shape[-1])
                    kl = (p_bar.clamp_min(1e-8) * (p_bar.clamp_min(1e-8) / uniform).log()).sum()
                    aux_loss = -args.moe_entropy_coef * gate_entropy + args.moe_balance_coef * kl

                total_loss = td_loss + aux_loss

                optimizer.zero_grad()
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(q_network.parameters(), 10.0)
                optimizer.step()

            if args.tau < 1.0:
                with torch.no_grad():
                    for tp, p in zip(target_network.parameters(), q_network.parameters()):
                        tp.data.lerp_(p.data, args.tau)
            else:
                if global_step % args.target_network_frequency == 0:
                    target_network.load_state_dict(q_network.state_dict())

            if global_step % 100 == 0:
                writer.add_scalar("losses/td_loss", td_loss.item(), global_step)
                if args.model_type == "moe":
                    writer.add_scalar("losses/aux_loss", aux_loss.item(), global_step)

    if args.save_model:
        suffix = "softmoe_penult" if args.model_type == "moe" else f"wide_x{args.wide_multiplier}"
        model_path = f"runs/{run_name}/{args.exp_name}.{suffix}.pt"
        os.makedirs(os.path.dirname(model_path), exist_ok=True)
        torch.save(q_network.state_dict(), model_path)
        print(f"model saved to {model_path}")

    envs.close()
    writer.close()
