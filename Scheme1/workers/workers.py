import numpy as np
import utils
from math import floor, sqrt, log
import tqdm 

class WorkerA:
    def __init__(self, params):
        np.random.seed(params['random_seed'])
        self.env, self.server, self.mechanism, self.profiler = utils.get_settings(params)
        T = int(params['env']['time_horizon'])   
        if params['echo']:
            self.iterator = tqdm.tqdm(range(T+1))
        else:
            self.iterator = range(T+1)

    def run(self):
        contexts = self.env.initialize()
        for t in self.iterator:
            record = dict()
            select_index, select_context  = self.server.decide(contexts)
            reward, pseudo_regret, next_contexts = self.env.step(select_index)
            noised_package = self.mechanism.pack(select_context, reward, self.server)
            temp = np.linalg.norm(self.server.theta)
            self.server.update(t, noised_package)
            contexts = next_contexts 
            self.profiler.record(t, reward, pseudo_regret, np.linalg.norm(self.server.theta - self.env.theta),temp)
            
        self.profiler.publish()

    def save(self, dist_file):
        self.profiler.df.to_csv(dist_file, index = False)
        
class WorkerB:
    def __init__(self, params):
        np.random.seed(params['random_seed'])
        self.env, self.server, self.mechanism, self.profiler = utils.get_settings(params)
        T = int(params['env']['time_horizon'])   
        if params['echo']:
            self.iterator = tqdm.tqdm(range(T+1))
        else:
            self.iterator = range(T+1)

    def run(self):
        contexts = self.env.initialize()
        # self.server.set_param(self.env.theta)
        for t in self.iterator:
            temp = np.linalg.norm(self.server.theta)
            record = dict()
            select_index, select_context  = self.server.decide(contexts)
            reward, pseudo_regret, next_contexts = self.env.step(select_index)
            noised_package = self.mechanism.pack(select_context, reward, self.server)
            self.server.update(t, noised_package)
            f_diff  = np.linalg.norm(noised_package['f_diff'])
            contexts = next_contexts 
            self.profiler.record(t, reward, pseudo_regret, np.linalg.norm(self.server.theta - self.env.theta), f_diff)
            
        self.profiler.publish()

    def save(self, dist_file):
        self.profiler.df.to_csv(dist_file, index = False)

        