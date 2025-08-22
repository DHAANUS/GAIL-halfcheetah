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
