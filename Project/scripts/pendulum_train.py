import gym
import os
import sys

root = os.path.dirname(os.getcwd())
sys.path.append(root)

from infrastructure.buffer import ReplayBuffer
from utils.util import build_network, pendulum_plot
from agents.ddpg_agent import DDPGAgent
from learners.ddpg_learner import DDPGLearner

buffer = ReplayBuffer(10**4)
env = gym.make('Pendulum-v0')
env.reset()

n_states = env.observation_space.shape[0]
n_actions = env.action_space.shape[0]

q_network = build_network(n_states+n_actions, 1, 2, 256)
p_network = build_network(n_states, n_actions, 2, 256, (-2, 2))
pg = DDPGAgent(q_network, p_network, 0.99, 0.01, 1e-3, 1e-4)

learner = DDPGLearner(env, buffer, pg, maximize=True, batch_size=128)
hist = learner.train(presample=10, noise_scale=2.0, episodes=10001, interval=10, plot=None, save_path="./", save_name="pendulum")

