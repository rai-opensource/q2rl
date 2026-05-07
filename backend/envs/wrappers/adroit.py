import gym


class AdroitTerminalWrapper(gym.Wrapper):
    """
    The original adroit environment doesn't set the done signal when the goal is
    achieved. This wrapper sets the done signal when the episode is done,
    decided by if the reward is 4
    """

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        # this wrapper should be wrapped right after environment creation and before
        # the Trucation wrapper, so it returns a 4-tuple
        obs, reward, done, info = self.env.step(action)
        if info["goal_achieved"]:
            done = True
        return obs, reward, done, info
