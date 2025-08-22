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
