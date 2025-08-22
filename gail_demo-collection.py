
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
