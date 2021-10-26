import numpy as np

class Server:
    def __init__(self, params):
        self.T = params['env']['time_horizon']
        self.epsilon = params['mechanism']['epsilon']
        self.delta = params['mechanism']['delta']
        self.d = params['env']['dimension']
        self.env_s = params['env']['instance_variance']
        self.I = np.identity(self.d)
        
    def output(self):
        pass
    
