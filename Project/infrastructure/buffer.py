import numpy as np
from infrastructure.sumtree import SumTree


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


class PrioritizedBuffer:
    """
    A prioritized buffer based on sum-tree.
    """

    def __init__(self, max_length=10**6, eps=0.01, alpha=0.6, beta=0.4):
        self.tree = SumTree(max_length)

        self.eps = eps
        self.alpha = alpha
        self.beta  = beta
    
    def push(self, exp, p=None):
        if p is None:
            N = self.tree.capacity
            p = np.max(self.tree.tree[-N:])
        else:
            p = self._getPriority(p)
        
        if p < self.eps:
            p = self.eps
        self.tree.push(exp, p)

    def sample(self, n_samples):
        idxs = []
        samples = []
        ws = []
        seg = self.tree.total / n_samples

        N = self.tree.capacity
        norm = np.maximum(np.max(self.tree.tree[-N:]), 0)

        for i in range(n_samples):
            low = seg * i
            high= seg * (i+1)

            s = np.random.uniform(low, high)
            idx, p, data = self.tree.get(s)

            idxs.append(idx)
            samples.append(data)
            ws.append((p / norm)**(-self.beta))

        return idxs, samples, ws

    def update(self, idx, p):
        p = self._getPriority(p)
        self.tree.update(idx, p)
    
    def batchupdate(self, idxs, ps):
        for idx, p in zip(idxs, ps):
            self.update(idx, p)

    def _getPriority(self, p):
        ## modify p with exponents alpha
        return (p + self.eps) ** self.alpha