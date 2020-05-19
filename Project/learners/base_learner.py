class BaseLearner:
    """
    A base learner class to control training process.
    """

    def __init__(self, env, buffer, agent, maximize=False):
        """
        Initialize learner object.

        Args:
            env: the environment;
            buffer: a ReplayBuffer instance;
            agent: a DPPGAgent instance;
        """

        self.env = env
        self.buffer = buffer
        self.agent = agent
        

    def train(self):
        raise NotImplementedError
    
    def sample_trail(self, policy, batch_size=128, episode_length=1000, add_to_buffer=False, train=False):
        """
        Sample a trail with given policy.

        Args:
            policy: the policy function at each time step;
            batch_size: sample batch size from the replay buffer;
            episode_length: the maximum length of each episode;
            add_to_buffer: whether to add experience to replay buffer;
            train: whether to train the agent at each time step.
        
        Returns:
            rewards: the reward sequence;
            traj: the system state sequence;
            us: the control input sequence.
        """

        terminal = False
        x = self.env.reset()
        
        rewards = []
        traj = [x]
        us = []
        i = 0

        while i < episode_length:
            u = policy(x)

            x_next, reward, terminal, _ = self.env.step(u)
            
            if add_to_buffer:
                self.buffer.push((x, u, reward, x_next, terminal))
            
            if train:
                exps = self.buffer.sample(batch_size)
                self.agent.train(exps)

            x = x_next

            rewards.append(reward)
            traj.append(x_next)
            us.append(u)

            i += 1

            if terminal:
                break
        
        return rewards, traj, us