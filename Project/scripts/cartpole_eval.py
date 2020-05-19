import numpy as np
import tensorflow as tf
import gym
from gym import wrappers
import os
import matplotlib.pyplot as plt


## environment
env = gym.make('CartPole-v0')
env = wrappers.Monitor(env, '../animations/', force=True)
env.reset()

## GPU configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)


def play(env, policy):
    x = env.reset()
    terminal = False
    rewards = []

    while not terminal:
        env.render()
        u = policy.predict(x.reshape([1, -1]))
        u = np.argmax(u)
        x, r, terminal, _ = env.step(u)

        rewards.append(r)
    
    return np.sum(rewards)


# DQN
policy = tf.keras.models.load_model("../models/DQN_q_network.h5")
play(env, policy)

## Double DQN
policy = tf.keras.models.load_model("../models/DoubleDQN_q_network.h5")
play(env, policy)

## Prioritized Experience Replay
policy = tf.keras.models.load_model("../models/PrioritizedDQN_q_network.h5")
play(env, policy)

## Deuling DQN
policy = tf.keras.models.load_model("../models/DeulDQN_q_network.h5")
play(env, policy)