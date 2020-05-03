from agents.base_agent import BaseAgent

import os
import numpy as np
import tensorflow as tf

class DDPGAgent(BaseAgent):
    """
    A DDPG agent implementation for continuous cases.
    """

    def __init__(self, q_network, p_network, alpha, tau, q_lr, p_lr):
        self.q_network = q_network
        self.p_network = p_network
        self.alpha = alpha

        self.q_optim = tf.keras.optimizers.Adam(lr=q_lr)
        self.p_optim = tf.keras.optimizers.Adam(lr=p_lr)

        self.target_q_network = tf.keras.models.clone_model(self.q_network)
        self.target_p_network = tf.keras.models.clone_model(self.p_network)

        self.target_q_network.set_weights(self.q_network.get_weights())
        self.target_p_network.set_weights(self.p_network.get_weights())

        self.tau = tau
    
    def __call__(self, x):
        """
        Call the p-network to get action with given state.
        """

        flatten = False
        if len(x.shape) == 1:
            x = np.array([x])
            flatten = True

        u = self.p_network.predict(tf.convert_to_tensor(x))
        if flatten:
            u = u.flatten()

        return u
    

    def train(self, exps):
        """
        Train the network with given experiences.
        """

        xs, us, cs, x_nexts, terminals = [], [], [], [], []
        for exp in exps:
            x, u, c, x_next, terminal = exp
            xs.append(x)
            us.append(u)
            cs.append(c)
            x_nexts.append(x_next)
            if terminal:
              terminals.append(1.0)
            else:
              terminals.append(0.0)
        
        xs = tf.convert_to_tensor(np.array(xs), dtype=tf.float32)
        us = tf.convert_to_tensor(np.array(us), dtype=tf.float32)
        cs = tf.convert_to_tensor(np.array(cs), dtype=tf.float32)
        x_nexts = tf.convert_to_tensor(np.array(x_nexts), dtype=tf.float32)
        terminals = tf.convert_to_tensor(np.array(terminals), dtype=tf.float32)

        ## update q-network
        with tf.GradientTape() as tape:
            q = self.q_network(tf.concat([xs, us], 1))
            q = tf.reshape(q, [-1])
            u_nexts = self.target_p_network(x_nexts)
            q_= self.target_q_network(tf.concat([x_nexts, u_nexts], 1))
            q_= tf.reshape(q_, [-1])
            q_target = cs + self.alpha * q_ * (1-terminals)

            L = tf.losses.MSE(q_target, q)

        q_grads = tape.gradient(L, self.q_network.trainable_weights)
        self.q_optim.apply_gradients(grads_and_vars=zip(q_grads, self.q_network.trainable_weights))

        ## update p-network
        with tf.GradientTape() as tape:
            J = self.q_network(tf.concat([xs, self.p_network(xs)], 1))
            J = tf.reduce_mean(J)
        
        p_grads = tape.gradient(J, self.p_network.trainable_weights)
        self.p_optim.apply_gradients(grads_and_vars=zip(p_grads, self.p_network.trainable_weights))

        ## update target networks
        self.update_target_networks()
    

    def update_target_networks(self):
        """
        Update target networks.
        """

        source_params = self.q_network.trainable_weights + self.p_network.trainable_weights
        target_params = self.target_q_network.trainable_weights+self.target_p_network.trainable_weights

        for target, source in  zip(target_params, source_params):
            target.assign(target * (1-self.tau) + source * self.tau)


    def save(self, path, name):
        """
        Save the networks to path.
        """

        self.q_network.save(os.path.join(path, f"{name}_q_network.h5"))
        self.p_network.save(os.path.join(path, f"{name}_p_network.h5"))

        print(f"Models are saved to {path}")