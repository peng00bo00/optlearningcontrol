import gym
from gym import wrappers
import os
import numpy as np
import tensorflow as tf


## GPU configuration
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

env = gym.make('InvertedDoublePendulum-v2')
env = wrappers.Monitor(env, '../animations/', force=True)
x = env.reset()
env.render()

policy = tf.keras.models.load_model("../models/DoubleInvertedPendulum_p_network.h5")
print("Model is loaded!")

i = 0
terminal = False

while (not terminal) and (i < 1000):
    env.render()

    u = policy.predict(x.reshape((1, -1)))
    x, r, terminal, _ = env.step(u)
    i += 1

    print(f"Step: {i}, Control: {u[0, 0]}, Reward: {r}")

env.close()
print("Finished!")