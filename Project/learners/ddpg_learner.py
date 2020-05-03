import numpy as np


class DDPGLearner:
    """
    A learner class to control DDPG training process.
    """
    def __init__(self, env, buffer, agent, maximize=False, batch_size=128):
        """
        Initialize learner object.

        Args:
        env: the environment;
        buffer: a ReplayBuffer instance;
        agent: a DPPGAgent instance;
        maximize: whether to maximize the reward sequence. If set True the learner would try to maximize
                  the environment rewards, or the learner would minimize the cost;
        batch_size: sample batch size from the replay buffer;
        """

        self.env = env
        self.buffer = buffer
        self.agent = agent
        self.maximize = maximize
        self.batch_size = batch_size
    
    def sample_trail(self, policy, add_to_buffer=False, train=False):
        """
        Sample a trail with given policy.

        Args:
            policy: the policy function at each time step;
            add_to_buffer: whether to add experience to replay buffer;
            train: whether to train the agent at each time step.
        
        Returns:
            costs: the cost sequence;
            traj: the system state sequence;
            us: the control input sequence.
        """

        terminal = False
        x = self.env.reset()
        
        costs = []
        traj = [x]
        us = []

        while not terminal:
            u = policy(x)

            x_next, cost, terminal, _ = self.env.step(u)

            ## whether to maximize the rewards
            if self.maximize:
                cost = -cost
            
            if add_to_buffer:
                self.buffer.push((x, u, cost, x_next, terminal))
            
            if train:
                exps = self.buffer.sample(self.batch_size)
                self.agent.train(exps)

            x = x_next

            costs.append(cost)
            traj.append(x_next)
            us.append(u)
        
        return costs, traj, us

    
    def train(self, presample=10, noise_scale=1.0, episodes=100, interval=10, plot=None, save_path=None, save_name="DDPG"):
        """
        Start the training process.

        Args:
            presample: the number of presampling (with random policy);
            noise_scale: the std of gaussian noises:
            episodes: the number of training episodes;
            interval: the interval for test current agent;
            plot: a plot function;
            save_path: path to save model;
            save_name: name of the model;

        Returns:
            hist: a dictionary contains training history
        """

        n_actions = self.env.action_space.shape[0]

        print("Start pre-sampling with random policy...")
        for t in range(presample):
            random_policy = lambda x: 2*noise_scale*np.random.rand(n_actions) - noise_scale
            self.sample_trail(random_policy, add_to_buffer=True)
            
        print(f"Pre-sampling finished!")

        hist = {"costs": [], "states": [], "controls": []}
        for t in range(episodes):
            noise_policy = lambda x: self.agent(x) + np.random.normal(0, noise_scale*np.exp(-t), n_actions)
            self.sample_trail(noise_policy, add_to_buffer=True, train=True)
            
            ## report every interval steps
            if t % interval == 0:
                costs, traj, us = self.sample_trail(self.agent)
                ## add trajectory to hist
                hist["costs"].append(costs)
                hist["states"].append(traj)
                hist["controls"].append(us)

                print(f"Episode: {t}, Mean Cost: {np.mean(costs)}")

                if plot is not None:
                    plot(traj=traj, us=us, costs=costs)
                
                if save_path is not None:
                    self.agent.save(save_path, save_name)
    
        print("Training finished!")

        return hist
