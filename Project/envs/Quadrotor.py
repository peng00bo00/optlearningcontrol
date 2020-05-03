from envs.env import BaseEnv
import numpy as np

class Quadrotor(BaseEnv):
    """
    A quadrotor environment.
    """

    def __init__(self, m, I, r, g=9.81, dt=0.01, target=np.array([0, 1, 0, 0, 0, 0]), Q=np.eye(6), R=0.01*np.eye(2)):
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
        self.thre = target @ Q @ target * 2
        self.thre = self.thre.flatten()

    def _step(self, a):
        ## current cost
        dx = self.target - self.x
        cost = dx.T @ self.Q @ dx + a.T @ self.R @ a
        cost = cost.flatten()[0]

        terminal = False

        dt = self.dt
        g = self.g
        m = self.m
        I = self.I
        r = self.r

        u1 = a[0]
        u2 = a[1]

        ## next state
        x, y, theta, u, v, omega = self.x

        x_next = x + dt * u
        y_next = y + dt * v
        if y_next < 0:
            y_next = 0
            terminal = True
        theta_next = theta + dt * omega
        u_next = u - dt/m*(u1+u2)*np.sin(theta)
        v_next = v + dt/m*(u1+u2)*np.cos(theta) - g*dt
        omega_next = omega + dt/I*(u1-u2)*r

        self.t += dt

        ## check if the simulation ends
        if (theta_next > np.pi / 2) or (theta_next < -np.pi / 2):
            terminal = True
        if self.t > 10:
            terminal = True
        if x_next < -1 or x_next > 1 or y_next > 2:
            terminal = True 
        # if cost > self.thre:
        #     terminal = True

        self.x = np.array([x_next, y_next, theta_next, u_next, v_next, omega_next])

        return self.x, cost, terminal

    def _reset(self):
        ## state = [x, y, theta, u, v, omega]
        self.x = np.zeros(6)
        self.t = 0.

        return self.x


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