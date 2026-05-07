from gym import Wrapper


class TruncationWrapper(Wrapper):
    """d4rl only supports the old gym API, where env.step returns a 4-tuple without
    the truncated signal. Here we explicity expose the truncated signal."""

    def __init__(self, env):
        super().__init__(env)

    def reset(self):
        s = self.env.reset()
        return s, {}

    def step(self, a):
        s, r, done, info = self.env.step(a)
        if "TimeLimit.truncated" in info:
            truncated = True
            done = False
        else:
            truncated = False
        return s, r, done, truncated, info
