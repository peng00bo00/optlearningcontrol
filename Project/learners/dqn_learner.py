import numpy as np
from learners.base_learner import BaseLearner


class DQNLearner(BaseLearner):
    """
    A learner class to control DQN training process.
    """

    def train(self, presample=10, batch_size=128, episodes=1000, episode_length=1000, interval=10, C=10, plot=None, save_path=None, save_name="DQN"):
        """
        Start the training process.

        Args:
            presample: the number of presampling (with random policy);
            episodes: the number of training episodes;
            episode_length: the maximum length of each episode;
            interval: the interval for test current agent;
            C: update target network every C steps;
            plot: a plot function;
            save_path: path to save model;
            save_name: name of the model;

        Returns:
            hist: a dictionary contains training history
        """

        print("Start pre-sampling with random policy...")
        random_policy = lambda x: self.env.action_space.sample()
        for t in range(presample):
            self.sample_trail(random_policy, batch_size, episode_length, add_to_buffer=True)
        
        print("Pre-sampling finished!")

        hist = {"rewards": [], "states": [], "controls": [], "episode length": []}
        for t in range(episodes):
            policy = self.agent
            ## explore with epsilon-greedy
            policy = lambda x: self.env.action_space.sample() if np.random.rand() < np.exp(-t) else self.agent(x)
            self.sample_trail(policy, batch_size, episode_length, add_to_buffer=True, train=True)

            ## Update target networks
            if t % C == 0:
                self.agent.update_target_networks()
            
            ## report every interval steps
            if t % interval == 0:
                rewards, traj, us = self.sample_trail(self.agent.predict, batch_size, episode_length)
                ## add trajectory to hist
                hist["rewards"].append(rewards)
                hist["states"].append(traj)
                hist["controls"].append(us)
                hist["episode length"].append(len(us))

                print(f"Episode: {t}, Episode Length: {len(us)}, Mean Reward: {np.mean(rewards)}, Total Reward: {np.sum(rewards)}")

                if plot is not None:
                    plot(traj=traj, us=us, rewards=rewards)
                
                if save_path is not None:
                    self.agent.save(save_path, save_name)
    
        print("Training finished!")

        return hist


class PrioritizedDQNLearner(DQNLearner):
    """
    A learner class to control DQN training process with prioritized replay buffer.
    """
    
    def sample_trail(self, policy, batch_size=128, episode_length=1000, add_to_buffer=False, train=False):
        """
        Sample a trail with given policy. Override this function for prioritized replay buffer.

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
                idxs, exps, ws = self.buffer.sample(batch_size)
                delta = self.agent.train(exps, ws)
                ## update priorities
                self.buffer.batchupdate(idxs, delta)

            x = x_next

            rewards.append(reward)
            traj.append(x_next)
            us.append(u)

            i += 1

            if terminal:
                break
        
        return rewards, traj, us