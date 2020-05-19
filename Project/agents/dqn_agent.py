from agents.base_agent import BaseAgent

import os
import numpy as np
import tensorflow as tf

class DQNAgent(BaseAgent):
    """
    A DQN agent implementation for discrete cases.
    """

    def __init__(self, q_network, alpha, lr):
        self.q_network = q_network
        self.alpha = alpha

        self.optim = tf.keras.optimizers.Adam(lr=lr)

        self.target_q_network = tf.keras.models.clone_model(self.q_network)
        self.target_q_network.set_weights(self.q_network.get_weights())

    
    def __call__(self, x):
        """
        Call the q-values to get action with given state. Only used in training.
        """

        flatten = False
        if len(x.shape) == 1:
            x = np.array([x])
            flatten = True

        u = self.q_network(tf.convert_to_tensor(x))
        u = tf.math.argmax(u, axis=1)
        u = u.numpy()

        if flatten:
            u = u[0]

        return u
    
    def predict(self, x):
        """
        Call the q-values to get action with given state. This is different from direct call, 
        since BN have different behaviours in training/ evaluation.
        """

        flatten = False
        if len(x.shape) == 1:
            x = np.array([x])
            flatten = True

        u = self.q_network.predict(tf.convert_to_tensor(x))
        u = tf.math.argmax(u, axis=1)
        u = u.numpy()

        if flatten:
            u = u[0]

        return u
    
    def train(self, exps):
        """
        Train the network with given experiences.
        """

        xs, us, rs, x_nexts, terminals = [], [], [], [], []
        for exp in exps:
            x, u, r, x_next, terminal = exp
            xs.append(x)
            us.append(u)
            rs.append(r)
            x_nexts.append(x_next)
            if terminal:
              terminals.append(1.0)
            else:
              terminals.append(0.0)
        
        xs = tf.convert_to_tensor(np.array(xs), dtype=tf.float32)
        us = tf.convert_to_tensor(np.array(us), dtype=tf.int32)
        rs = tf.convert_to_tensor(np.array(rs), dtype=tf.float32)
        x_nexts = tf.convert_to_tensor(np.array(x_nexts), dtype=tf.float32)
        terminals = tf.convert_to_tensor(np.array(terminals), dtype=tf.float32)

        ## update q-network
        with tf.GradientTape() as tape:
            q = self.q_network(xs)
            q = tf.math.reduce_sum(q * tf.one_hot(us, q.shape[1]), axis=1)
            q_= self.target_q_network(x_nexts)
            q_= tf.math.reduce_max(q_, axis=1)
            q_target = rs + self.alpha * q_ * (1-terminals)

            L = tf.losses.MSE(q_target, q)

        q_grads = tape.gradient(L, self.q_network.trainable_weights)
        self.optim.apply_gradients(grads_and_vars=zip(q_grads, self.q_network.trainable_weights))
    

    def update_target_networks(self):
        """
        Update target networks.
        """

        for target, source in  zip(self.target_q_network.trainable_weights, self.q_network.trainable_weights):
            target.assign(source)

    def save(self, path, name):
        """
        Save the networks to path.
        """

        self.q_network.save(os.path.join(path, f"{name}_q_network.h5"))

        print(f"Models are saved to {path}")


class DoubleDQNAgent(DQNAgent):
    """
    A Double DQN agent implementation for discrete cases.
    """

    def train(self, exps):
        """
        Train the network with given experiences.
        """

        xs, us, rs, x_nexts, terminals = [], [], [], [], []
        for exp in exps:
            x, u, r, x_next, terminal = exp
            xs.append(x)
            us.append(u)
            rs.append(r)
            x_nexts.append(x_next)
            if terminal:
              terminals.append(1.0)
            else:
              terminals.append(0.0)
        
        xs = tf.convert_to_tensor(np.array(xs), dtype=tf.float32)
        us = tf.convert_to_tensor(np.array(us), dtype=tf.int32)
        rs = tf.convert_to_tensor(np.array(rs), dtype=tf.float32)
        x_nexts = tf.convert_to_tensor(np.array(x_nexts), dtype=tf.float32)
        terminals = tf.convert_to_tensor(np.array(terminals), dtype=tf.float32)

        ## update q-network
        with tf.GradientTape() as tape:
            q = self.q_network(xs)
            q = tf.math.reduce_sum(q * tf.one_hot(us, q.shape[1]), axis=1)
            # q_= self.target_q_network(tf.concat([x_nexts, u_nexts], 1))
            q_= self.target_q_network(x_nexts)
            # q_= tf.math.reduce_max(q_, axis=1)
            u_= tf.math.argmax(self.q_network(x_nexts), axis=1)
            q_= tf.math.reduce_sum(q_* tf.one_hot(u_, q_.shape[1]), axis=1)
            q_target = rs + self.alpha * q_ * (1-terminals)

            L = tf.losses.MSE(q_target, q)

        q_grads = tape.gradient(L, self.q_network.trainable_weights)
        self.optim.apply_gradients(grads_and_vars=zip(q_grads, self.q_network.trainable_weights))


class DeulDQNAgent(DQNAgent):
    pass


class PrioritizedDQNAgent(DQNAgent):  
    def train(self, exps, ws):
        """
        Train the network with given experiences.
        """

        xs, us, rs, x_nexts, terminals = [], [], [], [], []
        for exp in exps:
            x, u, r, x_next, terminal = exp
            xs.append(x)
            us.append(u)
            rs.append(r)
            x_nexts.append(x_next)
            if terminal:
              terminals.append(1.0)
            else:
              terminals.append(0.0)
        
        xs = tf.convert_to_tensor(np.array(xs), dtype=tf.float32)
        us = tf.convert_to_tensor(np.array(us), dtype=tf.int32)
        rs = tf.convert_to_tensor(np.array(rs), dtype=tf.float32)
        x_nexts = tf.convert_to_tensor(np.array(x_nexts), dtype=tf.float32)
        terminals = tf.convert_to_tensor(np.array(terminals), dtype=tf.float32)

        ws = tf.convert_to_tensor(np.array(ws), dtype=tf.float32)

        ## update q-network
        with tf.GradientTape() as tape:
            q = self.q_network(xs)
            q = tf.math.reduce_sum(q * tf.one_hot(us, q.shape[1]), axis=1)
            # q_= self.target_q_network(tf.concat([x_nexts, u_nexts], 1))
            q_= self.target_q_network(x_nexts)
            # q_= tf.math.reduce_max(q_, axis=1)
            u_= tf.math.argmax(self.q_network(x_nexts), axis=1)
            q_= tf.math.reduce_sum(q_* tf.one_hot(u_, q_.shape[1]), axis=1)
            q_target = rs + self.alpha * q_ * (1-terminals)

            L = tf.math.reduce_mean(tf.math.pow(q_target - q, 2) * ws)

        delta = q_target.numpy()
        delta = np.abs(delta)
        q_grads = tape.gradient(L, self.q_network.trainable_weights)
        self.optim.apply_gradients(grads_and_vars=zip(q_grads, self.q_network.trainable_weights))

        return delta
