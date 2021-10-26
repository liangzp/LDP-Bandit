class BaseProfiler:

    def __init__(self, params):
        self._name = params['name']

    def record(self, t, reward, regret, est_error):
        self.ts.append(t) 
        self.rewards.append(reward)
        self.pseudo_regrets.append(regret)
        self.est_errors.append(est_error)

    def publish(self):
        pass
