import numpy as np


class ReplayBuffer:
    """
    A replay buffer to store experiences.
    """

    def __init__(self, max_length=10**6):
        self.cache = []
        self.max_length = max_length
    
    def __len__(self):
        return len(self.cache)
    
    def push(self, exp):
        """
        Add experience to the buffer.
        """

        if len(self) == self.max_length:
            self.cache = self.cache[1:]
        
        self.cache.append(exp)
    
    def sample(self, n_samples):
        """
        Sample n experiences from the buffer.
        """

        N = len(self)

        assert n_samples <= N, "The number of samples should be lower than length of buffer."

        idx = np.random.choice(N, n_samples)
        samples = []

        for i in idx:
            samples.append(self.cache[i])
        
        return samples
