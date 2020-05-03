class BaseEnv:
    """
    An abstract environment class.
    """

    def reset(self):
        return self._reset()

    def step(self, u):
        return self._step(u)
    