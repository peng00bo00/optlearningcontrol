import gym
from gym import wrappers
import os
import numpy as np
import tensorflow as tf

env = gym.make('Pendulum-v0')
env = wrappers.Monitor(env, '../animations/', force=True)
x = env.reset()
env.render()

policy = tf.keras.models.load_model("../models/pendulum_p_network.h5")
print("Model is loaded!")

for i in range(200):
    env.render()

    u = policy(x.reshape((1, 3)))
    x, r, _, _ = env.step(u)

    print(f"Step: {i}, Control: {u[0, 0]}, Reward: {r}")

env.close()
print("Finished!")
