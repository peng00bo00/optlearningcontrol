from envs.env import BaseEnv
import numpy as np

class CartPole(BaseEnv):
    """
    A quadrotor environment.
    """

    def __init__(self, mc, mp, l, g=9.81, dt=0.01, target=np.array([0, np.pi, 0, 0]), Q=np.eye(4), R=0.01*np.eye(1)):
        self.mc = mc
        self.mp = mp
        self.l = l

        self.g = g
        self.dt = dt

        ## target = [x, theta, v, omega]
        self.target = target

        self.Q = Q
        self.R = R

        self.reset()

        self.observation_space = np.zeros(4)
        self.action_space = np.zeros(1)

    def _step(self, u):
        ## current cost
        dx = self.target - self.x
        cost = dx.T @ self.Q @ dx + u.T @ self.R * u
        cost = cost.flatten()[0]

        terminal = False

        dt = self.dt
        g = self.g
        mc = self.mc
        mp = self.mp
        l = self.l
        u = u[0]

        ## next state
        x, theta, v, omega = self.x

        x_next = x + dt * v
        theta_next = theta + dt * omega

        dv = (u+mp*np.sin(theta)*(l*omega**2 + g*np.cos(theta))) / (mc + mp *np.sin(theta)**2)
        domega = (-u*np.cos(theta)-mp*l*omega**2*np.cos(theta)*np.sin(theta)-(mc+mp)*g*np.sin(theta)) / (l*(mc + mp *np.sin(theta)**2))

        v_next = v + dt * dv
        omega_next = omega + dt * domega

        self.t += dt

        ## check if the simulation ends
        if self.t > 10:
            terminal = True
        # if cost < 1e-3 or cost > 1e5:
        #     terminal = True
        # if np.abs(theta_next - np.pi) > np.pi / 2:
        #     terminal = True
        if x_next < -3 or x > 3:
            terminal = True
        
        if theta_next > 2 * np.pi:
            theta_next -= 2*np.pi
        elif theta_next < 0:
            theta_next += 2*np.pi

        self.x = np.array([x_next, theta_next, v_next, omega_next])

        return self.x, cost, terminal, {}

    def _reset(self):
        ## state = [x, theta, v, omega]
        # self.x = np.zeros(4)
        self.x = np.array([0, np.pi+np.random.normal(scale=0.1), 0, 0])
        self.t = 0.

        return self.x