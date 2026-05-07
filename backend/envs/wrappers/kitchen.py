import gym


class KitchenTerminalWrapper(gym.Wrapper):
    """
    The original kitchen environment doesn't set the done signal when the episode is
    successfully completed. This wrapper sets the done signal when the episode is done,
    decided by if the reward is 4
    """

    def __init__(self, env):
        super().__init__(env)

    def step(self, action):
        # this wrapper should be wrapped right after environment creation and before
        # the Trucation wrapper, so it returns a 4-tuple
        obs, reward, done, info = self.env.step(action)
        if reward == 4 or done:
            done = True
        return obs, reward, done, info
    
    def render(self, mode='human'):
        return self.env.render(mode=mode)
