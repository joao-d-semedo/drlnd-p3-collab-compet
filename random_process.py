import numpy as np

class OrnsteinUhlenbeckProcess:
    def __init__(self, size, std=0.2, theta=0.15, dt=1.0, x0=None):
        self.size = size
        self.std = std
        self.theta = theta
        self.dt = dt
        self.x0 = x0
        
        self.mu = 0.0
        self.reset_states()

    def sample(self):
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt + self.std * np.sqrt(
            self.dt) * np.random.randn(*self.size)
        self.x_prev = x
        return x

    def reset_states(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros(self.size)

