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
