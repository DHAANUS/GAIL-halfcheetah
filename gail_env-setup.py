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
