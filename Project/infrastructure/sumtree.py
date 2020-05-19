import numpy as np


class SumTree:
    """
    A sum-tree data structure for prioritized replay buffer.
    """

    def __init__(self, capacity):
        """
        Initialize sum-tree with given capacity.
        """

        ## self.data is used to store data
        self.data = np.zeros(capacity, dtype=object)
        ## self.tree is used to record weights
        self.tree  = np.zeros(2*capacity-1)
        self.capacity = capacity

        ## self.position is used to record data index
        self.position = 0
    
    def _propogate(self, idx, change):
        """
        Update the weights recursively from bottom up.
        """

        parent = (idx - 1) // 2

        self.tree[parent] += change

        if parent != 0:
            self._propogate(parent, change)

    def _retrieve(self, idx, s):
        """
        Traverse the tree recursively from top down.
        Return the leaf index in the tree.
        """
        left  = 2*idx + 1
        right = 2*idx + 2

        ## return idx if it's already leaf
        if left >= len(self.tree):
            return idx
        
        ## traverse recursively
        if s <= self.tree[left]:
            return self._retrieve(left, s)
        else:
            return self._retrieve(right, s-self.tree[left])

    def push(self, data, p):
        """
        Add a new node into the tree.
        """
        idx = self.position + self.capacity - 1

        self.data[self.position] = data
        self.update(idx, p)

        self.position += 1
        if self.position >= self.capacity:
            self.position = 0

    def update(self, idx, p):
        change = p - self.tree[idx]

        self.tree[idx] = p
        self._propogate(idx, change)

    def get(self, s):
        idx = self._retrieve(0, s)
        dataIdx = idx - (self.capacity - 1)

        return idx, self.tree[idx], self.data[dataIdx]
    
    @property
    def total(self):
        return self.tree[0]