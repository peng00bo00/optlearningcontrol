import numpy as np
import tensorflow as tf
import gym

from infrastructure.buffer import ReplayBuffer, PrioritizedBuffer
from utils.util import build_network, build_duel_network, cartpole_plot
from agents.dqn_agent import DQNAgent, DoubleDQNAgent,  DeulDQNAgent, PrioritizedDQNAgent
from learners.dqn_learner import DQNLearner, PrioritizedDQNLearner


## environment
env = gym.make('CartPole-v0')
env.reset()

## GPU configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

n_states = env.observation_space.shape[0]
n_actions = env.action_space.n

## parameter settings
BUFFER_SIZE = 10**4
EPISODES = 1001
EPISODE_LENGTH = 200
DISCOUNT = 0.99
LR = 1e-3
PRESAMPLE = 10
BATCH_SIZE = 128
C = 10
INTERVAL = 10

## DQN
env.reset()
buffer = ReplayBuffer(BUFFER_SIZE)
dqn_q_network = build_network(n_states, n_actions, 2, 200)

agent = DQNAgent(dqn_q_network, DISCOUNT, LR)
learner = DQNLearner(env, buffer, agent)

dqn_hist = learner.train(presample=PRESAMPLE, batch_size=BATCH_SIZE, episodes=EPISODES, episode_length=EPISODE_LENGTH, interval=INTERVAL, C=C, save_path="./", save_name="DQN")

## Double DQN
env.reset()
buffer = ReplayBuffer(BUFFER_SIZE)
double_q_network = build_network(n_states, n_actions, 2, 200)

agent = DoubleDQNAgent(double_q_network, DISCOUNT, LR)
learner = DQNLearner(env, buffer, agent)

double_hist = learner.train(presample=PRESAMPLE, batch_size=BATCH_SIZE, episodes=EPISODES, episode_length=EPISODE_LENGTH, interval=INTERVAL, C=C, save_path="./", save_name="DoubleDQN")

## Prioritized Experience Replay
env.reset()
buffer = PrioritizedBuffer(BUFFER_SIZE)
prioritized_q_network = build_duel_network(n_states, n_actions, 2, 200)

agent = PrioritizedDQNAgent(prioritized_q_network, DISCOUNT, LR)
learner = PrioritizedDQNLearner(env, buffer, agent)

prioritized_hist = learner.train(presample=PRESAMPLE, batch_size=BATCH_SIZE, episodes=EPISODES, episode_length=EPISODE_LENGTH, interval=INTERVAL, C=C, save_path="./", save_name="PrioritizedDQN")

## Deuling DQN
env.reset()
buffer = ReplayBuffer(BUFFER_SIZE)
duel_q_network = build_duel_network(n_states, n_actions, 2, 200)

agent = DeulDQNAgent(duel_q_network, DISCOUNT, LR)
learner = DQNLearner(env, buffer, agent)

duel_hist = learner.train(presample=PRESAMPLE, batch_size=BATCH_SIZE, episodes=EPISODES, episode_length=EPISODE_LENGTH, interval=INTERVAL, C=C, save_path="./", save_name="DeulDQN")