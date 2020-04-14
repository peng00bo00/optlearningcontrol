from envs.env import Env
import numpy as np

class Quadrotor(Env):
    """
    A quadrotor environment.
    """

    def __init__(self, m, I, r, g=9.81, dt=0.01, target=np.array([10, 10, 0, 0, 0, 0]), Q=np.eye(6), R=np.eye(2)):
        self.m = m
        self.I = I
        self.r = r

        self.g = g
        self.dt = dt

        ## target = [x, y, theta, u, v, omega]
        self.target = target

        self.Q = Q
        self.R = R

        self.reset()

    def _step(self, u):
        ## current cost
        dx = self.target - self.x
        cost = dx.T @ self.Q @ dx + u.T @ self.R @ u
        cost = cost.flatten()[0]

        dt = self.dt
        g = self.g
        m = self.m
        I = self.I
        r = self.r

        u1 = u[0]
        u2 = u[1]

        ## next state
        x, y, theta, u, v, omega = self.x

        x_next = x + dt * u
        y_next = y + dt * v
        theta_next = theta + dt * omega
        u_next = u - dt/m*(u1+u2)*np.sin(theta)
        v_next = v + dt/m*(u1+u2)*np.cos(theta) - g*dt
        omega_next = omega + dt/I*(u1-u2)*r

        self.x = np.array([x_next, y_next, theta_next, u_next, v_next, omega_next])

        return cost

    def _reset(self):
        ## state = [x, y, theta, u, v, omega]
        self.x = np.zeros(6)


if __name__ == "__main__":
    m = 1.0
    I = 1.0
    r = 1.0

    u = np.zeros(2)
    quadrotor = Quadrotor(m, I, r)

    print(quadrotor.x)
    quadrotor.step(u)
    print(quadrotor.x)
    quadrotor.step(u)
    print(quadrotor.x)
    quadrotor.step(u)
    print(quadrotor.x)

    print("Debug succeed!")