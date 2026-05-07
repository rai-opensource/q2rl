"""
Policy evaluation utilities.

Provides functions for evaluating learned policies in gym environments,
including episode statistics collection and optional video recording.
"""
from collections import defaultdict
from typing import Dict, Optional, List

import gym
import numpy as np
import wandb
import cv2
import collections
import robomimic.utils.obs_utils as ObsUtils

def get_robomimic_obs(obs):
    """
    Convert observation dict to RoboMimic observation format.
    
    Args:
        obs: State observation array with EEF, gripper, and object info.
        
    Returns:
        Processed RoboMimic observation dictionary.
    """
    robomimic_obs = collections.OrderedDict()
    robomimic_obs["robot0_eef_pos"] = obs[:3]
    robomimic_obs["robot0_eef_quat"] = obs[3:7]
    robomimic_obs["robot0_gripper_qpos"] = obs[7:9]
    robomimic_obs["object"] = obs[9:]
    robomimic_obs = ObsUtils.process_obs_dict(robomimic_obs)
    return robomimic_obs

def get_robomimic_obs_img(obs):
    """
    Convert image + state observation to RoboMimic format.
    
    Args:
        obs: Dictionary with 'state', 'front', and 'wrist' keys.
        
    Returns:
        Processed RoboMimic observation dictionary.
    """
    robomimic_obs = collections.OrderedDict()
    robomimic_obs["robot0_eef_pos"] = obs["state"][:3]
    robomimic_obs["robot0_eef_quat"] = obs["state"][3:7]
    robomimic_obs["robot0_gripper_qpos"] = obs["state"][7:9]

    robomimic_obs["agentview_image"] = obs["front"]
    robomimic_obs["robot0_eye_in_hand_image"] = obs["wrist"]
    robomimic_obs = ObsUtils.process_obs_dict(robomimic_obs)
    return robomimic_obs

def flatten(d, parent_key="", sep="."):
    """
    Flatten nested dictionary.
    
    Args:
        d: Nested dictionary.
        parent_key: Prefix for keys (used recursively).
        sep: Separator for nested keys (default '.').
        
    Returns:
        Flattened dictionary with dot-separated keys.
    """
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, "items"):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """
    Add dictionary values to dict of lists.
    
    Args:
        dict_of_lists: Dictionary mapping keys to lists.
        single_dict: Dictionary to add values from.
    """
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def evaluate(
    policy_fn, env: gym.Env, num_episodes: int, clip_action: float = np.inf
) -> Dict[str, float]:
    """
    Run policy in environment and collect episode statistics.
    
    Args:
        policy_fn: Callable taking observation and returning action.
        env: Gym environment.
        num_episodes: Number of episodes to run.
        clip_action: Maximum absolute action value (default inf).
        
    Returns:
        Dictionary of aggregated statistics (means across episodes).
    """
    stats = defaultdict(list)
    for _ in range(num_episodes):
        observation, info = env.reset()
        add_to(stats, flatten(info))
        done = False
        while not done:
            action = policy_fn(observation)
            action = np.clip(action, -clip_action, clip_action)
            observation, _, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            add_to(stats, flatten(info))
        add_to(stats, flatten(info, parent_key="final"))

    for k, v in stats.items():
        stats[k] = np.mean(v)
    return stats

def evaluate_and_record_videos_baseline(
    policy_fn, env: gym.Env, num_episodes: int, clip_action: float = np.inf,
    record_video: bool = False, wandb_logger: Optional[object] = None, step_number: int = 0
) -> Dict[str, float]:
    """
    Evaluate policy and optionally record a video to wandb.

    Args:
        policy_fn: Callable taking observation and returning action.
        env: Gym environment.
        num_episodes: Number of episodes to run.
        clip_action: Maximum absolute action value (default inf).
        record_video: Whether to record and log video frames.
        wandb_logger: wandb run object for logging (required if record_video=True).
        step_number: Training step number for wandb logging.

    Returns:
        Tuple of (stats_dict, trajectories_list).
    """
    trajectories = []
    stats = defaultdict(list)
    episode_frames = []
    for _ in range(num_episodes):
        trajectory = defaultdict(list)
        observation, info = env.reset()
        add_to(stats, flatten(info))
        done = False
        while not done:
            if record_video:
                frame = env.render(mode='rgb_array')
                episode_frames.append(frame)
            action = policy_fn(observation)
            action = np.clip(action, -clip_action, clip_action)
            next_observation, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            transition = dict(
                observations=observation,
                next_observations=next_observation,
                actions=action,
                rewards=r,
                dones=done,
                infos=info,
                masks=1 - terminated,
            )
            add_to(trajectory, transition)
            add_to(stats, flatten(info))
            observation = next_observation
        add_to(stats, flatten(info, parent_key="final"))
        trajectories.append(trajectory)

    episode_frames = np.array(episode_frames)
    # Log videos to wandb if logger is provided
    if record_video and wandb_logger:
        video_caption = f"Episode"       
        # Ensure frames are uint8 and in range [0, 255]
        if episode_frames.dtype != np.uint8:
            if episode_frames.max() <= 1.0:
                episode_frames = (episode_frames * 255).astype(np.uint8)
            else:
                episode_frames = episode_frames.astype(np.uint8)

        episode_frames =  np.transpose(episode_frames,(0,3,1,2))
        wandb_video = wandb.Video(
            episode_frames,
            fps=30,
            format= "mp4",
            caption=video_caption
        )
        wandb_logger.log({"eval_video": wandb_video}, step=step_number)

    for k, v in stats.items():
        stats[k] = np.mean(v)
    return stats, trajectories

def evaluate_and_record_videos_baseline_robsuite(
    policy_fn, bc_fn, env: gym.Env, num_episodes: int, step_number: int, clip_action: float = np.inf, record_video: bool = False,
    video_fps: int = 30,
    video_format: str = "mp4",
    max_video_length: Optional[int] = None,
    wandb_logger: Optional[object] = None,
) -> Dict[str, float]:
    """
    Evaluate policy on a RoboSuite environment and optionally record video to wandb.

    Args:
        policy_fn: Callable taking observation and returning (action, ...).
        bc_fn: Behavioral cloning function (not currently used in loop body).
        env: Gym-wrapped RoboSuite environment.
        num_episodes: Number of episodes to run.
        step_number: Training step number for wandb logging and video caption.
        clip_action: Maximum absolute action value (default inf).
        record_video: Whether to record and log video frames.
        video_fps: Frames per second for the recorded video.
        video_format: Video format string (e.g. 'mp4').
        max_video_length: Unused maximum video length parameter.
        wandb_logger: wandb run object for logging (required if record_video=True).

    Returns:
        Tuple of (stats_dict, None).
    """
    trajectories = []
    stats = defaultdict(list)
    episode_frames = []
    success = 0
    average_return = 0
    average_traj_length = 0

    for _ in range(num_episodes):
        observation, info = env.reset()
        done = False
        while not done:
            if record_video:
                frame = env.render(mode='rgb_array')
                episode_frames.append(frame)
            action = policy_fn(observation)[0]
            action = np.clip(action, -clip_action, clip_action)
            next_observation, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            if truncated:
                success += 1
            average_return += r
            average_traj_length += 1
            add_to(stats, flatten(info))
            observation = next_observation
        add_to(stats, flatten(info, parent_key="final"))
    
    episode_frames = np.array(episode_frames)
    # Log videos to wandb if logger is provided
    if record_video and wandb_logger:
        video_caption = f"Episode {step_number}, Length: {average_traj_length}"       
        # Ensure frames are uint8 and in range [0, 255]
        if episode_frames.dtype != np.uint8:
            if episode_frames.max() <= 1.0:
                episode_frames = (episode_frames * 255).astype(np.uint8)
            else:
                episode_frames = episode_frames.astype(np.uint8)

        episode_frames =  np.transpose(episode_frames,(0,3,1,2))
        wandb_video = wandb.Video(
            episode_frames,
            fps=video_fps,
            format=video_format,
            caption=video_caption
        )
        wandb_logger.log({"eval_video": wandb_video}, step=step_number)
    
    stats["success_rate"] = success / num_episodes
    stats["average_return"] = average_return / num_episodes
    stats["average_traj_length"] = average_traj_length / num_episodes
    return stats, None

def evaluate_and_record_videos_q2rl(
    policy_fn, 
    env: gym.Env, 
    num_episodes: int, 
    step_number: int,
    clip_action: float = np.inf,
    record_video: bool = False,
    video_frequency: int = 1,
    video_fps: int = 30,
    video_format: str = "mp4",
    max_video_length: Optional[int] = None,
    wandb_logger: Optional[object] = None,
) -> Dict[str, float]:
    """
    Evaluate policy and optionally record videos of episodes.
    
    Args:
        policy_fn: Function that takes observation and returns action
        env: Gym environment 
        num_episodes: Number of episodes to evaluate
        clip_action: Clip actions to [-clip_action, clip_action]
        record_video: Whether to record videos
        video_frequency: Record video every N episodes (1 = record all)
        video_fps: Frames per second for video
        video_format: Video format ("mp4", "gif", "webm", "avi")
        max_video_length: Maximum number of frames per video
        wandb_logger: WandB logger instance to log videos
        video_prefix: Prefix for video names in wandb
        
    Returns:
        Dictionary of evaluation statistics
    """
    stats = defaultdict(list)
    recorded_videos = []
    episode_frames = []
    trajectories = []
    for episode_idx in range(num_episodes):
        trajectory = defaultdict(list)
        observation, info = env.reset()
        add_to(stats, flatten(info))
        done = False
        step_count = 0
        
        while not done:
            # Record frame if video recording is enabled
            if record_video:
                frame = env.render(mode='rgb_array')
                episode_frames.append(frame)
            action, sample_info = policy_fn(observation)
            action = np.clip(action, -clip_action, clip_action)
            next_observation, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            add_to(stats, flatten(info))
            if sample_info:
                add_to(stats, flatten(sample_info))
            transition = dict(
                observations=observation,
                next_observations=next_observation,
                actions=action,
                rewards=r,
                dones=done,
                infos=info,
                masks=1 - terminated,
            )
            add_to(trajectory, transition)
            observation = next_observation
            step_count += 1
                
        add_to(stats, flatten(info, parent_key="final"))
        trajectories.append(trajectory)
        
    episode_frames = np.array(episode_frames)
    # Log videos to wandb if logger is provided
    if record_video and wandb_logger:
        video_caption = f"Episode {episode_idx}, Length: {step_count}"       
        # Ensure frames are uint8 and in range [0, 255]
        if episode_frames.dtype != np.uint8:
            if episode_frames.max() <= 1.0:
                episode_frames = (episode_frames * 255).astype(np.uint8)
            else:
                episode_frames = episode_frames.astype(np.uint8)

        episode_frames =  np.transpose(episode_frames,(0,3,1,2))
        wandb_video = wandb.Video(
            episode_frames,
            fps=video_fps,
            format=video_format,
            caption=video_caption
        )
        wandb_logger.log({"eval_video": wandb_video}, step=step_number)

    for k, v in stats.items():
        stats[k] = np.mean(v)
    return stats, trajectories

def evaluate_and_record_videos_q2rl_robosuite(
    policy_fn, 
    bc_fn,
    env: gym.Env, 
    num_episodes: int, 
    step_number: int,
    clip_action: float = np.inf,
    record_video: bool = True,
    video_frequency: int = 1,
    video_fps: int = 30,
    video_format: str = "mp4",
    max_video_length: Optional[int] = None,
    wandb_logger: Optional[object] = None,
    img_obs: bool = False,
) -> Dict[str, float]:
    """
    Evaluate policy and optionally record videos of episodes.
    
    Args:
        policy_fn: Function that takes observation and returns action
        env: Gym environment 
        num_episodes: Number of episodes to evaluate
        clip_action: Clip actions to [-clip_action, clip_action]
        record_video: Whether to record videos
        video_frequency: Record video every N episodes (1 = record all)
        video_fps: Frames per second for video
        video_format: Video format ("mp4", "gif", "webm", "avi")
        max_video_length: Maximum number of frames per video
        wandb_logger: WandB logger instance to log videos
        video_prefix: Prefix for video names in wandb
        
    Returns:
        Dictionary of evaluation statistics
    """
    stats = defaultdict(list)
    episode_frames = []
    success = 0
    average_return = 0
    average_traj_length = 0
    for episode_idx in range(num_episodes):
        observation, info = env.reset()
        add_to(stats, flatten(info))
        done = False
        step_count = 0
        
        while not done:
            # Record frame if video recording is enabled
            if img_obs:
                bc_action, bc_log_prob, entropy = bc_fn(ob=get_robomimic_obs_img(observation), log_prob=True)
            else:
                bc_action, bc_log_prob, entropy = bc_fn(ob=get_robomimic_obs(observation), log_prob=True)

            action, sample_info = policy_fn(observation, bc_action=bc_action, bc_log_prob=bc_log_prob, bc_entropy=entropy)
            action = np.clip(action, -clip_action, clip_action)
            next_observation, r, terminated, truncated, info = env.step(action)
            if record_video:
                frame = env.render(mode='rgb_array')
                if sample_info["bc_actions"] > 0:
                    border_color = (0, 255, 0)  # Green for BC action
                else:
                    border_color = (255, 0, 0)  # Red for policy action
                ##Add border 
                # Top and bottom
                frame[:5, :, :] = border_color
                frame[-5:, :, :] = border_color
                # Left and right
                frame[:, :5, :] = border_color
                frame[:, -5:, :] = border_color
                frame = np.ascontiguousarray(frame)
                # Add text on top
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.4
                font_color = (255, 255, 255)  # White
                thickness = 1
                # Position (x, y) -- (10, 30) near top left
                cv2.putText(frame, f"bc:{sample_info['bc_q']:.2f}", (10, 30), font, font_scale, font_color, thickness, cv2.LINE_AA)
                cv2.putText(frame, f"rl:{sample_info['rl_q']:.2f}", (10, 60), font, font_scale, font_color, thickness, cv2.LINE_AA)
                episode_frames.append(frame)
            done = terminated or truncated
            if truncated:
                success += 1
            average_return += r
            average_traj_length += 1
            add_to(stats, flatten(info))
            if sample_info:
                add_to(stats, flatten(sample_info))
            
            observation = next_observation
            step_count += 1          
        add_to(stats, flatten(info, parent_key="final"))
        
    episode_frames = np.array(episode_frames)
    if record_video and wandb_logger:
        video_caption = f"Episode {episode_idx}, Length: {step_count}"       
        # Ensure frames are uint8 and in range [0, 255]
        if episode_frames.dtype != np.uint8:
            if episode_frames.max() <= 1.0:
                episode_frames = (episode_frames * 255).astype(np.uint8)
            else:
                episode_frames = episode_frames.astype(np.uint8)

        episode_frames =  np.transpose(episode_frames,(0,3,1,2))
        wandb_video = wandb.Video(
            episode_frames,
            fps=video_fps,
            format=video_format,
            caption=video_caption
        )
        wandb_logger.log({"eval_video": wandb_video}, step=step_number)

    for k, v in stats.items():
        stats[k] = np.mean(v)
    stats["success_rate"] = success / num_episodes
    stats["average_return"] = average_return / num_episodes
    stats["average_traj_length"] = average_traj_length / num_episodes
    return stats, None


def evaluate_and_record_videos_ibrl(
    policy_fn, 
    bc_fn,
    env: gym.Env, 
    num_episodes: int, 
    step_number: int,
    clip_action: float = np.inf,
    record_video: bool = True,
    video_frequency: int = 1,
    video_fps: int = 30,
    video_format: str = "mp4",
    max_video_length: Optional[int] = None,
    wandb_logger: Optional[object] = None,
) -> Dict[str, float]:
    """
    Evaluate policy and optionally record videos of episodes.
    
    Args:
        policy_fn: Function that takes observation and returns action
        env: Gym environment 
        num_episodes: Number of episodes to evaluate
        clip_action: Clip actions to [-clip_action, clip_action]
        record_video: Whether to record videos
        video_frequency: Record video every N episodes (1 = record all)
        video_fps: Frames per second for video
        video_format: Video format ("mp4", "gif", "webm", "avi")
        max_video_length: Maximum number of frames per video
        wandb_logger: WandB logger instance to log videos
        video_prefix: Prefix for video names in wandb
        
    Returns:
        Dictionary of evaluation statistics
    """
    stats = defaultdict(list)
    recorded_videos = []
    episode_frames = []
    trajectories = []
    for episode_idx in range(num_episodes):
        trajectory = defaultdict(list)
        observation, info = env.reset()
        add_to(stats, flatten(info))
        done = False
        step_count = 0
        
        while not done:
            # Record frame if video recording is enabled
            if record_video:
                frame = env.render(mode='rgb_array')
                episode_frames.append(frame)
            
            bc_action = bc_fn(observation, argmax=True)
            action, diff = policy_fn(observation, bc_actions=bc_action)
            action = np.clip(action, -clip_action, clip_action)
            next_observation, r, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            add_to(stats, flatten(info))
            transition = dict(
                observations=observation,
                next_observations=next_observation,
                actions=action,
                rewards=r,
                dones=done,
                infos=info,
                masks=1 - terminated,
            )
            add_to(trajectory, transition)
            observation = next_observation
            step_count += 1
           
        add_to(stats, flatten(info, parent_key="final"))
        trajectories.append(trajectory)
        
    episode_frames = np.array(episode_frames)
    # Log videos to wandb if logger is provided
    if record_video and wandb_logger:
        video_caption = f"Episode {episode_idx}, Length: {step_count}"       
        # Ensure frames are uint8 and in range [0, 255]
        if episode_frames.dtype != np.uint8:
            if episode_frames.max() <= 1.0:
                episode_frames = (episode_frames * 255).astype(np.uint8)
            else:
                episode_frames = episode_frames.astype(np.uint8)

        episode_frames =  np.transpose(episode_frames,(0,3,1,2))
        wandb_video = wandb.Video(
            episode_frames,
            fps=video_fps,
            format=video_format,
            caption=video_caption
        )
        wandb_logger.log({"eval_video": wandb_video}, step=step_number)

    for k, v in stats.items():
        stats[k] = np.mean(v)
    return stats, trajectories

def evaluate_and_record_videos_ibrl_robosuite(
    policy_fn, 
    bc_fn,
    env: gym.Env, 
    num_episodes: int, 
    step_number: int,
    clip_action: float = np.inf,
    record_video: bool = True,
    video_frequency: int = 1,
    video_fps: int = 30,
    video_format: str = "mp4",
    max_video_length: Optional[int] = None,
    wandb_logger: Optional[object] = None,
    img_obs: bool = False,
) -> Dict[str, float]:
    """
    Evaluate policy and optionally record videos of episodes.
    
    Args:
        policy_fn: Function that takes observation and returns action
        env: Gym environment 
        num_episodes: Number of episodes to evaluate
        clip_action: Clip actions to [-clip_action, clip_action]
        record_video: Whether to record videos
        video_frequency: Record video every N episodes (1 = record all)
        video_fps: Frames per second for video
        video_format: Video format ("mp4", "gif", "webm", "avi")
        max_video_length: Maximum number of frames per video
        wandb_logger: WandB logger instance to log videos
        video_prefix: Prefix for video names in wandb
        
    Returns:
        Dictionary of evaluation statistics
    """
    stats = defaultdict(list)
    episode_frames = []
    trajectories = []
    success = 0
    average_return = 0
    average_traj_length = 0
    bc_action_used=0
    for episode_idx in range(num_episodes):
        trajectory = defaultdict(list)
        observation, info = env.reset()
        add_to(stats, flatten(info))
        done = False
        step_count = 0
        
        while not done:
            # Record frame if video recording is enabled
            if img_obs:
                bc_action = bc_fn(ob=get_robomimic_obs_img(observation))
            else:
                bc_action = bc_fn(ob=get_robomimic_obs(observation))
           
            action, diff = policy_fn(observation, bc_actions=bc_action)
            if diff <0:
                bc_action_used = bc_action_used + 1

            action = np.clip(action, -clip_action, clip_action)
            next_observation, r, terminated, truncated, info = env.step(action)
            
            #frame = cv2.resize(frame, (512, 512))
            if record_video:
                frame = env.render(mode='rgb_array')
                
                if diff < 0:
                    border_color = (0, 255, 0)  # Green for BC action
                else:
                    border_color = (255, 0, 0)  # Red for policy action
                ##Add border 
                # Top and bottom
                frame[:5, :, :] = border_color
                frame[-5:, :, :] = border_color
                # Left and right
                frame[:, :5, :] = border_color
                frame[:, -5:, :] = border_color
                frame = np.ascontiguousarray(frame)
                episode_frames.append(frame)
            done = terminated or truncated
            if truncated:
                success += 1
            average_return += r
            average_traj_length += 1
            add_to(stats, flatten(info))
            transition = dict(
                observations=observation,
                next_observations=next_observation,
                actions=action,
                rewards=r,
                dones=done,
                infos=info,
                masks=1 - terminated,
            )
            add_to(trajectory, transition)
            observation = next_observation
            step_count += 1
                
        add_to(stats, flatten(info, parent_key="final"))
        trajectories.append(trajectory)
        
    episode_frames = np.array(episode_frames)
    # Log videos to wandb if logger is provided
    if record_video and wandb_logger:
        video_caption = f"Episode {episode_idx}, Length: {step_count}"       
        # Ensure frames are uint8 and in range [0, 255]
        if episode_frames.dtype != np.uint8:
            if episode_frames.max() <= 1.0:
                episode_frames = (episode_frames * 255).astype(np.uint8)
            else:
                episode_frames = episode_frames.astype(np.uint8)

        episode_frames =  np.transpose(episode_frames,(0,3,1,2))
        wandb_video = wandb.Video(
            episode_frames,
            fps=video_fps,
            format=video_format,
            caption=video_caption
        )
        wandb_logger.log({"eval_video": wandb_video}, step=step_number)

    stats["success_rate"] = success / num_episodes
    stats["average_return"] = average_return / num_episodes
    stats["average_traj_length"] = average_traj_length / num_episodes
    stats["bc_actions"] = bc_action_used / (average_traj_length)
    return stats, trajectories

