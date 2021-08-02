import numpy as np


class OUActionNoise():
    """Initialize parameters and noise process."""
    #  mu: mean for the noise, sigma: standart deviation, dt: time parameter, 
    #  x0: starting value
    def __init__(self, mu, sigma=0.2, theta=0.3, dt=1e-2, x0=None):
        self.theta = theta
        self.mu = mu
        self.sigma = sigma
        self.dt = dt
        self.x0 = x0
        self.reset()

    def __call__(self):
        #  noise = OUActionNoise()
        #  current_noise = noise
        x = self.x_prev + self.theta * (self.mu - self.x_prev) * self.dt +  \
            self.sigma * np.sqrt(self.dt) * np.random.normal(size=self.mu.shape)
        self.x_prev = x
        return x

    def reset(self):
        self.x_prev = self.x0 if self.x0 is not None else np.zeros_like(self.mu)
