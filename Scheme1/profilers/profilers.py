from profilers import BaseProfiler
import pandas as pd

class Profiler(BaseProfiler):
    def __init__(self, params):
        super().__init__(params)
        self.pseudo_regret = 0
        self.echo_freq = int(params['echo_freq'])
        
        self.thetas = list()
        self.contexts = list()
        self.rewards = list()
        self.pseudo_regrets = list()
        self.ts = list()
        self.est_errors = list()
        self.est_norm = list()
  
    def publish(self):
        df = dict() 
        df['rewards'] = self.rewards
        df['cum_regrets'] = self.pseudo_regrets
        df['time'] = self.ts
        df['estimation_error'] = self.est_errors
        df['estimation_norm'] = self.est_norm
        df = pd.DataFrame(df)
        self.df = df 
    
    def record(self, t, reward, regret, est_error, est_norm):
        self.pseudo_regret += regret
        if not (t % self.echo_freq):
            self.ts.append(t) 
            self.rewards.append(reward)
            self.pseudo_regrets.append(self.pseudo_regret)
            self.est_errors.append(est_error)
            self.est_norm.append(est_norm)
