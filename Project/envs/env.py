class Env:
    """
    An abstract environment class.
    """

    def reset(self):
        self._reset()

    def step(self, u):
        return self._step(u)