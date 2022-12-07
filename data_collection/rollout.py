"""
Script for collecting random rollouts.
"""

from pathlib import Path
import logging

from tqdm import tqdm
import numpy as np
from numpy.random import default_rng

from calvin_env.envs.play_table_env import get_env
from calvin_env.io_utils.data_recorder import DataRecorder
from calvin_env.utils.utils import count_frames, get_episode_lengths, set_egl_device, to_relative_action

log = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format='%(message)s')

def save_ep_lens(episode_lengths, num_prev_episodes):
    if num_prev_episodes > 0:
        previous_ep_lens = np.load("ep_lens.npy")
        episode_lengths = np.concatenate((previous_ep_lens, episode_lengths))
    np.save("ep_lens.npy", episode_lengths)
    end_ids = np.cumsum(episode_lengths) - 1
    start_ids = [0] + list(end_ids + 1)[:-1]
    ep_start_end_ids = list(zip(start_ids, end_ids))
    np.save("ep_start_end_ids.npy", ep_start_end_ids)

def save_step(counter, rgbs, depths, actions, robot_obs, scene_obs, cam_names, **additional_infos):
    rgb_entries = {f"rgb_{cam_name}": rgbs[f"rgb_{cam_name}"] for i, cam_name in enumerate(cam_names)}
    depths_entries = {f"depth_{cam_name}": depths[f"depth_{cam_name}"] for i, cam_name in enumerate(cam_names)}
    if actions[-1] == 0:
        actions[-1] = -1
    np.savez_compressed(
        f"episode_{counter:07d}.npz",
        actions=actions,
        rel_actions=to_relative_action(actions, robot_obs),
        robot_obs=robot_obs,
        scene_obs=scene_obs,
        **rgb_entries,
        **depths_entries,
        **additional_infos,
    )


def state_to_action(info):
    """
    save action as [tcp_pos, tcp_orn_quaternion, gripper_action]
    """
    tcp_pos = info["robot_info"]["tcp_pos"]
    tcp_orn = info["robot_info"]["tcp_orn"]
    gripper_action = info["robot_info"]["gripper_action"]
    action = np.concatenate([tcp_pos, tcp_orn, [gripper_action]])
    return action

def worker_run(env, load_dir, proc_num, start_frame, stop_frame, episode_index):
    log.info(f"[{proc_num}] Starting worker {proc_num}")
    # set_egl_device(0)

    log.info(f"[{proc_num}] Entering Loop")
    frame_counter = 0
    rgbs, depths, actions, robot_obs, scene_obs, = (
        [],
        [],
        [],
        [],
        [],
    )
    for frame in range(start_frame, stop_frame):
        file_path = Path(load_dir) / f"{frame:012d}.pickle"
        state_ob, done, info = env.reset_from_storage(file_path)
        action = state_to_action(info)
        robot_obs.append(state_ob["robot_obs"])
        scene_obs.append(state_ob["scene_obs"])

        # action is robot state of next frame
        if frame_counter > 0:
            actions.append(action)
        frame_rgbs, frame_depths = env.get_camera_obs()
        rgbs.append(frame_rgbs)
        depths.append(frame_depths)
        # for terminal states save current robot state as action
        frame_counter += 1
        log.debug(f"episode counter {episode_index} frame counter {frame_counter} done {done}")

        if frame_counter > 1:
            save_step(
                episode_index,
                rgbs.pop(0),
                depths.pop(0),
                actions.pop(0),
                robot_obs.pop(0),
                scene_obs.pop(0),
                cam_names=[cam.name for cam in env.cameras],
            )
            episode_index += 1
            if done:
                frame_counter = 0
                rgbs, depths, actions, robot_obs, scene_obs = [], [], [], [], []

        log.debug(f"[{proc_num}] Rendered frame {frame}")

    # assert done

    env.close()
    log.info(f"[{proc_num}] Finishing worker {proc_num}")

def main():

    rng = default_rng()
    datadir = "/home/basiri/Dev/calvin_env/data_collection/task_D_D"
    env = get_env(Path(datadir), show_gui=False)
    recorder = DataRecorder(env=env, record_fps=30.0, enable_tts=False)

    # --- record random agent's rollout ---
    N = 100
    max_length = 10
    action_mean = np.zeros(6)
    action_cov = np.diag([0.5]*6)
    actions = rng.multivariate_normal(action_mean, action_cov, N)
    gripper_actions = rng.choice([-1,1], N)
    for i, (action, gripper) in tqdm(enumerate(zip(actions, gripper_actions))):
        combined = np.hstack((action, gripper))
        obs, _, _, info = env.step(combined)
        done = True if (i+1)%max_length == 0 or i==N-1 else False
        recorder.step(prev_vr_event=(), state_obs=obs, done=done, info=info)

    # --- turn recordings into .npz files ---
    n_process = 8
    recording_dir = "/home/basiri/Dev/calvin_env/data_collection"
    max_frames = count_frames(recording_dir)
    episode_lengths, render_start_end_ids = get_episode_lengths(recording_dir, max_frames)
    worker_run(env, recording_dir, 0, 0, max_frames, 0)
    save_ep_lens(episode_lengths, 0)

    # --- test what the images look like by showing them using matplotlib ---
    # did not using EGL mess things up? nah it seems okay
    # import matplotlib.pyplot as plt
    # data = np.load('episode_0000010.npz')
    # rgb = data['rgb_static']
    # fig, ax = plt.subplots()
    # ax.imshow(rgb)
    # fig.savefig("test.png")
    # plt.show()

if __name__ == "__main__":
    main()
