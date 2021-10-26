class Environment:
    def __init__(self, params):
        self.T = params['env']['time_horizon']
        self.d = params['env']['dimension']
        self.s = params['env']['reward_variance']
        self.k = params['env']['n_action'] 
        self.env_s = params['env']['instance_variance']
        
    def initialize(self):
        pass
    
    def step(self, a):
        pass
