class Mechanism:
    def __init__(self, args):
        self.epsilon = args['mechanism']['epsilon']
        self.delta = args['mechanism']['delta']
        self.d = args['env']['dimension']
        
    def perturb(self):
        pass
