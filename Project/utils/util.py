import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt


## get this from https://github.com/openai/baselines/blob/master/baselines/ddpg/noise.py
class ActionNoise(object):
    def reset(self):
        pass

class OrnsteinUhlenbeckActionNoise(ActionNoise): # or replace 'ActionNoise' with 'object'
    def __init__(self, mu, sigma, theta=.15, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)

    def __repr__(self):
        return 'OrnsteinUhlenbeckActionNoise(mu={}, sigma={})'.format(self.mu, self.sigma)


def build_network(n_inputs, n_outputs, n_layers, hidden=16, limits=None):
    """
    Build a neural network with given parameters.
    """

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(hidden, activation="relu", input_shape=(n_inputs, ), kernel_initializer='random_uniform'))

    for i in range(n_layers):
        model.add(tf.keras.layers.BatchNormalization())
        model.add(tf.keras.layers.Dense(hidden, activation="relu", kernel_initializer='random_uniform'))
    
    if limits is not None:
        low, high = limits
        center = (low + high) / 2
        scale = high - center

        model.add(tf.keras.layers.Dense(n_outputs, activation="tanh", kernel_initializer='random_uniform'))
        model.add(tf.keras.layers.Lambda(lambda x: scale * x + center))
    else:
        model.add(tf.keras.layers.Dense(n_outputs, kernel_initializer='random_uniform'))

    return model


def build_duel_network(n_inputs, n_outputs, n_layers, hidden=16):
    """
    Build a neural network with given parameters.
    """

    inputs = tf.keras.layers.Input(n_inputs)

    x = tf.keras.layers.BatchNormalization()(inputs)
    x = tf.keras.layers.Dense(hidden, activation="relu", kernel_initializer='random_uniform')(x)

    for i in range(n_layers-1):
        x = tf.keras.layers.BatchNormalization()(x)
        x = tf.keras.layers.Dense(hidden, activation="relu", kernel_initializer='random_uniform')(x)

    ## value
    value = tf.keras.layers.BatchNormalization()(x)
    value = tf.keras.layers.Dense(1, kernel_initializer='random_uniform')(value)

    ## advantage
    advantage = tf.keras.layers.BatchNormalization()(x)
    advantage = tf.keras.layers.Dense(n_outputs, kernel_initializer='random_uniform')(advantage)
    advantage = advantage - tf.math.reduce_mean(advantage, axis=1, keepdims=True)

    ## outputs
    outputs = value + advantage

    model = tf.keras.Model(inputs=[inputs], outputs=[outputs])

    return model

def build_p_network(n_inputs, action_bound=1):
    """
    Build policy network.
    """

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(400, activation=None, input_shape=(n_inputs, )))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Dense(300, activation=None))
    model.add(tf.keras.layers.BatchNormalization())
    model.add(tf.keras.layers.ReLU())

    model.add(tf.keras.layers.Dense(1, activation="tanh", kernel_initializer=tf.random_uniform_initializer(minval=-0.003, maxval=0.003)))
    model.add(tf.keras.layers.Lambda(lambda x: action_bound * x))

    return model


def build_q_network(n_states, n_actions):
    """
    Build value network.
    """

    state = tf.keras.layers.Input(n_states)
    n1 = tf.keras.layers.Dense(400, activation=None)(state)
    n1 = tf.keras.layers.BatchNormalization()(n1)
    n1 = tf.keras.layers.ReLU()(n1)
    n1 = tf.keras.layers.Dense(300, activation=None)(n1)

    action = tf.keras.layers.Input(n_actions)
    n2 = tf.keras.layers.Dense(300, activation=None)(action)

    net = tf.keras.layers.Add()([n1, n2])
    net = tf.keras.layers.BatchNormalization()(net)
    net = tf.keras.layers.ReLU()(net)

    out = tf.keras.layers.Dense(1, activation=None, kernel_initializer=tf.random_uniform_initializer(minval=-0.003, maxval=0.003))(net)

    model = tf.keras.Model(inputs=[state, action], outputs=[out])

    return model

def pendulum_plot(traj, us, costs):
    """
    Plot Pendulum states/ control/ cost sequence.
    """
    
    plt.plot([x[0] for x in traj], label="cos")
    plt.plot([x[1] for x in traj], label="sin")
    plt.plot([x[2] for x in traj], label="dtheta")
    plt.legend()
    plt.title("State Sequence")
    plt.show()

    plt.plot([u[0] for u in us], label="u")
    plt.ylim((-2, 2))
    plt.legend(loc="upper right")
    plt.title("Control Sequence")
    plt.show()

    plt.plot(costs, label="cost")
    plt.legend(loc="upper right")
    plt.title("Cost Sequence")
    plt.show()


def cartpole_plot(traj, us, rewards):
    plt.plot([x[0] for x in traj], label="Cart Position")
    plt.plot([x[1] for x in traj], label="Cart Velocity")
    plt.plot([x[2] for x in traj], label="Pole Angle")
    plt.plot([x[3] for x in traj], label="Pole Velocity At Tip")
    plt.legend()
    plt.ylim((-3, 3))
    plt.title("State Sequence")
    plt.show()

    plt.plot(us, label="u")
    plt.legend(loc="upper right")
    plt.title("Control Sequence")
    plt.show()

    plt.plot(rewards, label="cost")
    plt.legend(loc="upper right")
    plt.title("Reward Sequence")
    plt.show()