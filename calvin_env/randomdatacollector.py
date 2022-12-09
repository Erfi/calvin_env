"""
Script for collecting random rollouts.
"""

import logging

import hydra
from tqdm import tqdm
import numpy as np
from numpy.random import default_rng

from calvin_env.io_utils.data_recorder import DataRecorder

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')


@hydra.main(config_path="../conf", config_name="config_random_data_collection")
def main(cfg):
    env = hydra.utils.instantiate(cfg.env)
    rng = default_rng()
    recorder = DataRecorder(env=env, record_fps=cfg.recorder.record_fps, enable_tts=False)

    num_rollouts = cfg.num_rollouts
    rollout_length = cfg.rollout_length
    total_steps = num_rollouts * rollout_length
    action_mean = np.zeros(6)
    action_cov = np.diag([0.5]*6)
    actions = rng.multivariate_normal(action_mean, action_cov, total_steps)
    gripper_actions = rng.choice([-1,1], total_steps)
    for i, (action, gripper) in tqdm(enumerate(zip(actions, gripper_actions))):
        combined = np.hstack((action, gripper))
        obs, _, _, info = env.step(combined)
        done = True if (i+1)%rollout_length == 0 else False
        recorder.step(prev_vr_event=(), state_obs=obs, done=done, info=info)
        if done:
            env.reset()
    env.close()

if __name__ == "__main__":
    main()
