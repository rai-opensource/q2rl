import copy
from typing import OrderedDict

import gym
import numpy as np
from gym import spaces
import cv2


class RobosuiteImageWrapper(gym.core.Env):
    #metadata = {"render_modes": ["rgb_array"]}
    metadata = None
    render_mode = None

    def __init__(self, env, reward_scale, reward_bias, keys=None):
        self.env = env
        self.reward_scale = reward_scale
        self.reward_bias = reward_bias
        
        # set up observation and action spaces
        obs = self.env.reset()

        self.observation_space = self.getObservationSpace(obs)
        self.action_space = spaces.Box(low=float("-1"), high=float("1"),
                                            shape=(7,), dtype=np.float32)

    def getObservationSpace(self, obs):
        keys  = ["front", "state", "wrist"]
        observation_spec =  OrderedDict()
        spec = obs["agentview_image"]
        observation_spec["front"] = spaces.Box(low=np.iinfo(spec.dtype).min, high=np.iinfo(spec.dtype).max,
                                                shape=spec.shape, dtype=spec.dtype)
        
        spec = obs["robot0_eye_in_hand_image"]
        observation_spec["wrist"] = spaces.Box(low=np.iinfo(spec.dtype).min, high=np.iinfo(spec.dtype).max,
                                                shape=spec.shape, dtype=spec.dtype)
        
        spec = obs["robot0_eef_pos"]
     
        observation_spec["state"] = spaces.Box(low=float("-inf"), high=float("inf"),
                                                shape=(9,), dtype=spec.dtype)
            

        return spaces.Dict(observation_spec)
    def robosuite_obs2gym_obs(self, obs):
        new_obs = OrderedDict()
       
        new_obs["front"] = cv2.flip(obs["agentview_image"], 0)
        new_obs["wrist"] = cv2.flip(obs["robot0_eye_in_hand_image"], 0)
        new_obs["state"] = np.concatenate((obs["robot0_eef_pos"],obs["robot0_eef_quat"],obs["robot0_gripper_qpos"]), axis = -1)
        return new_obs

    def reset(self, seed=None, options=None):
        """
        Extends env reset method to return flattened observation instead of normal OrderedDict and optionally resets seed

        Returns:
            np.array: Flattened environment observation space after reset occurs
        """
        if seed is not None:
            if isinstance(seed, int):
                np.random.seed(seed)
            else:
                raise TypeError("Seed must be an integer type!")
        ob_dict = self.env.reset()

        return self.robosuite_obs2gym_obs(ob_dict), {}

    def step(self, action):
        """
        Extends vanilla step() function call to return flattened observation instead of normal OrderedDict.

        Args:
            action (np.array): Action to take in environment

        Returns:
            4-tuple:

                - (np.array) fslattened observations from the environment
                - (float) reward from the environment
                - (bool) episode ending after reaching an env terminal state
                - (bool) episode ending after an externally defined condition
                - (dict) misc information
        """
        action = np.clip(action, self.action_space.low, self.action_space.high)
       
        ob_dict, reward, terminated, info = self.env.step(action)
        reward = reward * self.reward_scale + self.reward_bias

        truncated = self.env._check_success()
        return self.robosuite_obs2gym_obs(ob_dict), reward, terminated, truncated, info

    def render(self, mode="rgb_array"):
        if mode == "rgb_array":
            rgb_array = self.env.sim.render(height=256, width=256, camera_name="frontview")[::-1]
            return rgb_array
        
    def is_success(self):
        self.env._check_success()

    def compute_reward(self, achieved_goal, desired_goal, info):
        """
        Dummy function to be compatible with gym interface that simply returns environment reward

        Args:
            achieved_goal: [NOT USED]
            desired_goal: [NOT USED]
            info: [NOT USED]

        Returns:
            float: environment reward
        """
        # Dummy args used to mimic Wrapper interface
        return self.env.reward()
