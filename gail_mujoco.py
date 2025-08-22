#!/usr/bin/env python3
import os
import argparse
import numpy as np
from pathlib import Path

os.environ.setdefault("MUJOCO_GL", "egl")
from imitation.util.networks import RunningNorm

import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.ppo import MlpPolicy
from stable_baselines3.common.evaluation import evaluate_policy
from stable_baselines3.common.vec_env import VecNormalize, VecVideoRecorder
from imitation.util.util import make_vec_env as imit_make_vec_env
from imitation.data.wrappers import RolloutInfoWrapper
from imitation.data import rollout, serialize
from imitation.algorithms.adversarial.gail import GAIL
from imitation.rewards.reward_nets import BasicRewardNet
from imitation.util.networks import RunningNorm

RUNS = Path("runs")
MODELS = RUNS / "models"
DEMOS = RUNS / "demos"
VIDEOS = RUNS / "videos"
STATS = RUNS / "venv_stats"
for p in [RUNS, MODELS, DEMOS, VIDEOS, STATS]:
    p.mkdir(parents=True, exist_ok=True)


def make_env(env_id: str, n_envs: int, seed: int, render_rgb=False):
    env_kwargs = {"render_mode": "rgb_array"} if render_rgb else {}
    venv = imit_make_vec_env(
        env_id,
        rng=np.random.default_rng(seed),
        n_envs=n_envs,
        env_make_kwargs=env_kwargs,
        post_wrappers=[lambda env, _: RolloutInfoWrapper(env)],
    )
    venv = VecNormalize(venv, norm_obs=True, norm_reward=False, clip_obs=10.0)
    return venv

from pathlib import Path

def _resolve_model_path(env_id: str, which: str) -> Path:
    name = f"{env_id}_{'ppo_expert' if which=='expert' else 'gail_policy'}"
    candidates = [
        MODELS / f"{name}.zip",
        MODELS / name,
    ]
    for c in candidates:
        if c.exists():
            return c if c.suffix == ".zip" else (c.with_suffix(".zip") if (c.with_suffix(".zip")).exists() else c)
    hits = list(MODELS.glob(f"{name}*.zip"))
    if hits:
        return hits[0]
    raise FileNotFoundError(f"Model not found: looked for {candidates} and {name}*.zip")

def train_expert(env_id: str, total_steps: int, seed: int, tb_tag: str):
    venv = make_env(env_id, n_envs=8, seed=seed)
    model = PPO(
        policy=MlpPolicy,
        env=venv,
        n_steps=2048,
        batch_size=64,
        learning_rate=3e-4,
        ent_coef=0.0,
        gamma=0.99,
        gae_lambda=0.95,
        n_epochs=10,
        verbose=1,
        seed=seed,
        tensorboard_log=str(RUNS),
    )
    model.learn(total_timesteps=total_steps, tb_log_name=f"{tb_tag}_expert")
    model.save(MODELS / f"{env_id}_ppo_expert.zip")
    venv.save(STATS / f"{env_id}_expert_vecnorm.pkl")
    venv.close()

def collect_demos(env_id: str, min_steps: int, min_episodes: int, seed: int):
    venv = make_env(env_id, n_envs=8, seed=seed)
    venv = VecNormalize.load(STATS / f"{env_id}_expert_vecnorm.pkl", venv)
    venv.training = False
    expert = PPO.load(MODELS / f"{env_id}_ppo_expert.zip", env=venv)

    sample_until = rollout.make_sample_until(
        min_timesteps=min_steps if min_steps else None,
        min_episodes=min_episodes if min_episodes else None,
    )
    demos = rollout.rollout(
        expert,
        venv,
        sample_until=sample_until,
        rng=np.random.default_rng(seed),
    )
    serialize.save(DEMOS / f"{env_id}_expert_trajs", demos)
    venv.close()

# def train_gail(env_id: str, total_steps: int, seed: int, tb_tag: str):
#     venv = make_env(env_id, n_envs=8, seed=seed)
#     # learner = PPO(
#     #     policy=MlpPolicy,
#     #     env=venv,
#     #     batch_size=64,
#     #     learning_rate=4e-4,
#     #     gamma=0.95,
#     #     n_epochs=5,
#     #     ent_coef=0.0,
#     #     seed=seed,
#     #     tensorboard_log=str(RUNS),
#     #     verbose=1,
#     # )
#     learner = PPO(
#           policy=MlpPolicy,
#           env=venv,
#           batch_size=256,        # was 64
#           n_steps=2048,          # longer rollouts help IL
#           n_epochs=10,           # was 5
#           learning_rate=3e-4,    # slightly smaller than 4e-4
#           gamma=0.99,            # back to standard
#           ent_coef=0.01,         # encourage exploration
#           seed=seed,
#           tensorboard_log=str(RUNS),
#           verbose=1,
#       )
#     demos = serialize.load(DEMOS / f"{env_id}_expert_trajs")

#     reward_net = BasicRewardNet(
#         observation_space=venv.observation_space,
#         action_space=venv.action_space,
#         normalize_input_layer=RunningNorm,
#     )

#     # gail = GAIL(
#     #     demonstrations=demos,
#     #     demo_batch_size=1024,
#     #     gen_replay_buffer_capacity=10_000,
#     #     n_disc_updates_per_round=8,
#     #     venv=venv,
#     #     gen_algo=learner,
#     #     reward_net=reward_net,
#     # )
#     gail = GAIL(
#     demonstrations=demos,
#     demo_batch_size=256,           # was 1024
#     gen_replay_buffer_capacity=50_000,
#     n_disc_updates_per_round=2,    # was 8 (too many -> saturation)
#     venv=venv,
#     gen_algo=learner,
#     reward_net=reward_net,
#     )

#     pre_mean, pre_std = evaluate_policy(learner, venv, n_eval_episodes=5, return_episode_rewards=False)
#     print(f"[GAIL] pre-train eval return ~ {pre_mean:.2f} ± {pre_std:.2f}")

#     gail.train(total_timesteps=total_steps)


#     learner.save(MODELS / f"{env_id}_gail_policy.zip")
#     venv.save(STATS / f"{env_id}_gail_vecnorm.pkl")
#     venv.close()
def train_gail(env_id: str, total_steps: int, seed: int, tb_tag: str):
    """
    GAIL training that:
      - uses the SAME VecNormalize stats as the expert (frozen) to avoid scaling mismatch,
      - weakens the discriminator so it doesn't saturate,
      - strengthens PPO a bit with more exploration,
      - optionally warm-starts the policy with BC (version-agnostic).
    """
    # --- 1) Env with expert VecNormalize stats (frozen) ---
    venv = make_env(env_id, n_envs=8, seed=seed)
    expert_stats = STATS / f"{env_id}_expert_vecnorm.pkl"
    if expert_stats.exists():
        venv = VecNormalize.load(expert_stats, venv)
        venv.training = False   # freeze stats to match demos
    else:
        print(f"[WARN] Missing expert stats at {expert_stats}; GAIL may be unstable")

    # --- 2) PPO generator: steadier updates + a bit more exploration ---
    policy_kwargs = dict(log_std_init=-2.0)  # helps reduce early action spikes
    learner = PPO(
        policy=MlpPolicy,
        env=venv,
        n_steps=2048,
        batch_size=256,
        n_epochs=10,
        learning_rate=3e-4,
        gamma=0.99,
        ent_coef=0.01,
        seed=seed,
        tensorboard_log=str(RUNS),
        verbose=1,
        policy_kwargs=policy_kwargs,
    )

    # --- 3) Load expert demos ---
    demos = serialize.load(DEMOS / f"{env_id}_expert_trajs")

    # --- 4) Smaller, smoother discriminator (avoid instant 99% acc) ---
    from torch import nn
    reward_net = BasicRewardNet(
        observation_space=venv.observation_space,
        action_space=venv.action_space,
        normalize_input_layer=RunningNorm,
        hid_sizes=(64,),         # downsize capacity
        activation=nn.Tanh,      # smoother than ReLU
    )

    gail = GAIL(
        demonstrations=demos,
        demo_batch_size=256,               # was 1024 → reduce
        gen_replay_buffer_capacity=50_000, # larger buffer
        n_disc_updates_per_round=1,        # was 8 → avoid saturation
        venv=venv,
        gen_algo=learner,
        reward_net=reward_net,
    )

    # --- 5) (Optional) BC warm-start, robust across imitation versions ---
    try:
        from imitation.algorithms import bc as bc_mod
        from imitation.data import types as types_mod

        def trajs_to_transitions(trajs):
            # Use library helper if present
            if hasattr(types_mod, "trajectories_to_transitions"):
                return types_mod.trajectories_to_transitions(trajs)
            # Manual conversion (older/newer versions)
            TransitionsCls = getattr(types_mod, "Transitions", None) or getattr(types_mod, "TransitionsMinimal", None)
            if TransitionsCls is None:
                raise RuntimeError("imitation.data.types missing a Transitions class")

            obs_list, acts_list, next_obs_list, dones_list = [], [], [], []
            for t in trajs:
                o = t.obs            # (T+1, obs_dim)
                a = t.acts           # (T, act_dim)
                term = getattr(t, "terminal", None)
                if term is not None:
                    d = np.zeros(len(a), dtype=bool); d[-1] = bool(term)
                else:
                    d_attr = getattr(t, "dones", None)
                    if d_attr is None:
                        d = np.zeros(len(a), dtype=bool); d[-1] = True
                    else:
                        d = np.asarray(d_attr, dtype=bool)
                obs_list.append(o[:-1]); acts_list.append(a); next_obs_list.append(o[1:]); dones_list.append(d)

            obs = np.concatenate(obs_list)
            acts = np.concatenate(acts_list)
            next_obs = np.concatenate(next_obs_list)
            dones = np.concatenate(dones_list)
            try:
                return TransitionsCls(obs=obs, acts=acts, next_obs=next_obs, dones=dones, infos=None)
            except TypeError:
                return TransitionsCls(obs=obs, acts=acts, next_obs=next_obs, dones=dones)

        transitions = trajs_to_transitions(demos)
        bc = bc_mod.BC(
            observation_space=venv.observation_space,
            action_space=venv.action_space,
            demonstrations=transitions,
            policy=learner.policy,   # warm-start the SAME policy PPO will use
        )
        bc.train(n_epochs=5)
        print("[BC] Warm-start complete (5 epochs).")
    except Exception as e:
        print(f"[BC] Skipping warm-start: {e}")

    # --- 6) Sanity check on wrapped eval (not true env reward) ---
    pre_mean, pre_std = evaluate_policy(learner, venv, n_eval_episodes=5, return_episode_rewards=False)
    print(f"[GAIL] pre-train eval (wrapped) ~ {pre_mean:.2f} ± {pre_std:.2f}")

    # --- 7) Train GAIL ---
    gail.train(total_timesteps=total_steps)

    # --- 8) Save policy + the (frozen) stats we trained with ---
    learner.save(MODELS / f"{env_id}_gail_policy.zip")
    venv.save(STATS / f"{env_id}_gail_vecnorm.pkl")
    venv.close()


def evaluate(env_id: str, which: str, episodes: int, seed: int, video_len: int):
    render_env = make_env(env_id, n_envs=1, seed=seed, render_rgb=True)

    # Attach saved normalization stats if present
    stats_file = STATS / f"{env_id}_{which}_vecnorm.pkl"
    if stats_file.exists():
        render_env = VecNormalize.load(stats_file, render_env)

    # Freeze stats and report TRUE env rewards
    if isinstance(render_env, VecNormalize):
        render_env.training = False
        render_env.norm_reward = False

    model_path = _resolve_model_path(env_id, which)
    model = PPO.load(model_path, env=render_env)

    mean_r, std_r = evaluate_policy(model, render_env, n_eval_episodes=episodes, return_episode_rewards=False)
    print(f"[{which.upper()}] eval return (TRUE reward) ~ {mean_r:.2f} ± {std_r:.2f} over {episodes} episodes")

    video_env = VecVideoRecorder(
        render_env,
        video_folder=str(Path("runs") / "videos"),
        record_video_trigger=lambda step: step == 0,
        video_length=video_len,
        name_prefix=f"{env_id}_{which}",
    )
    obs = video_env.reset()      # no [0]
    t = 0
    while t < video_len:
        action, _ = model.predict(obs, deterministic=True)
        obs, rewards, dones, infos = video_env.step(action)  # 4 returns for VecEnv
        if dones[0]:
            obs = video_env.reset()
        t += 1
    video_env.close()
    render_env.close()

def main():
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="cmd", required=True)

    parser.add_argument("--env-id", type=str, default="HalfCheetah-v4")
    parser.add_argument("--seed", type=int, default=42)

    p_exp = sub.add_parser("expert")
    p_exp.add_argument("--steps", type=int, default=800_000)
    p_exp.add_argument("--tb-tag", type=str, default="expert")

    p_dem = sub.add_parser("demos")
    p_dem.add_argument("--min-steps", type=int, default=200_000)
    p_dem.add_argument("--min-episodes", type=int, default=0)

    p_gail = sub.add_parser("gail")
    p_gail.add_argument("--steps", type=int, default=800_000)
    p_gail.add_argument("--tb-tag", type=str, default="gail")

    p_eval = sub.add_parser("eval")
    p_eval.add_argument("--which", type=str, choices=["expert", "gail"], default="gail")
    p_eval.add_argument("--episodes", type=int, default=10)
    p_eval.add_argument("--video-len", type=int, default=1000)

    args = parser.parse_args()

    if args.cmd == "expert":
        train_expert(args.env_id, args.steps, args.seed, args.tb_tag)
    elif args.cmd == "demos":
        collect_demos(args.env_id, args.min_steps, args.min_episodes, args.seed)
    elif args.cmd == "gail":
        train_gail(args.env_id, args.steps, args.seed, args.tb_tag)
    elif args.cmd == "eval":
        evaluate(args.env_id, args.which, args.episodes, args.seed, args.video_len)

if __name__ == "__main__":
    main()
