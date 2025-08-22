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
import os
os.environ["MUJOCO_GL"] = "egl"
!python gail_mujoco.py --env-id HalfCheetah-v4 expert --steps 800000
!python gail_mujoco.py --env-id HalfCheetah-v4 demos --min-steps 200000
!python gail_mujoco.py --env-id HalfCheetah-v4 gail --steps 800000
!python gail_mujoco.py --env-id HalfCheetah-v4 eval --which gail --episodes 10 --video-len 1000
