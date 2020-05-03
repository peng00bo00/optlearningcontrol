import tensorflow as tf
import matplotlib.pyplot as plt


def build_network(n_inputs, n_outputs, n_layers, hidden=16, limits=None):
    """
    Build a neural network with given parameters.
    """

    model = tf.keras.Sequential()
    model.add(tf.keras.layers.Dense(hidden, activation="relu", input_shape=(n_inputs, )))

    for i in range(n_layers):
        model.add(tf.keras.layers.Dense(hidden, activation="relu"))
    
    if limits is not None:
        low, high = limits
        center = (low + high) / 2
        scale = high - center

        model.add(tf.keras.layers.Dense(n_outputs, activation="tanh"))
        model.add(tf.keras.layers.Lambda(lambda x: scale * x + center))
    else:
        model.add(tf.keras.layers.Dense(n_outputs))

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