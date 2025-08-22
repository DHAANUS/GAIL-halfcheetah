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

