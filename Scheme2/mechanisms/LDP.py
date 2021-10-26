from mechanisms import Mechanism
from math import sqrt, log, exp, pi
import numpy as np
from scipy.special import gamma

class NMechanism(Mechanism):
    def __init__(self, params):
        super(NMechanism, self).__init__(params)

    def pack(self, select_context, reward, server):
        xx = select_context.dot(select_context.T)
        yx = reward*select_context
        
        noised_package = dict()
        noised_package['xx'] = xx
        noised_package['yx'] = yx
        return noised_package

class OLSMechanism(Mechanism):
    def __init__(self, params):
        super().__init__(params)
        self.sigma = 6*sqrt(2*log(2.5/self.delta))/self.epsilon

    def perturb(self):
        noise_covariance = np.random.normal(0, self.sigma, size=(self.d,self.d))
        for i in range(len(noise_covariance)):
            for j in range(i):
                noise_covariance[i][j] = noise_covariance[j][i]

        noise_reward = np.random.normal(0, self.sigma, size=(self.d,1))
        return noise_covariance, noise_reward

    def pack(self, select_context, reward, server):
        covariance = select_context.dot(select_context.T)
        yx = reward*select_context

        noise_covariance, noise_reward = self.perturb()

        noised_package = dict()
        noised_package['xx'] = covariance + noise_covariance
        noised_package['yx'] = yx + noise_reward
        return noised_package

class SGDLDPMechanism(Mechanism):
    def __init__(self, params):
        super().__init__(params)
        self.k = params['env']['n_action']
        self.d = params['env']['dimension']
        self.env_s = params['env']['instance_variance']
        self.B = (exp(self.epsilon) + 1)/(exp(self.epsilon) - 1) * sqrt(pi) / 2 * self.d * gamma((self.d - 1)/2 + 1)/ gamma(self.d/2 + 1)
        self.threshold = exp(self.epsilon)/(exp(self.epsilon)+1)
        
    def pack(self, select_context, reward, server):
        gradient = (server.transform(np.clip(select_context.T.dot(server.theta), -20, 20)) - reward)*select_context
        noised_package = dict()
        x = gradient
        x = x/sqrt(x.T.dot(x)) if np.random.uniform()> (1/2+sqrt(x.T.dot(x))/(2)) else -x/sqrt(x.T.dot(x))
        prob = np.random.uniform()
        while True:
            z = np.random.normal(0, self.env_s, (self.d, 1))
            z = z/sqrt(z.T.dot(z))*self.B
            if (((prob>self.threshold) and (z.T.dot(x)>0)) or ((prob<=self.threshold) and (z.T.dot(x)<=0))):
                break

        noised_package['gradient'] = z
        return noised_package

    
class GLMMechanism(Mechanism):
    def __init__(self, params):
        super().__init__(params)
        self.sigma = 6*sqrt(2*log(2.5/self.delta))/self.epsilon
        self.env_s = params['env']['instance_variance']
        self.B = (exp(self.epsilon) + 1)/(exp(self.epsilon) - 1) * sqrt(pi) / 2 * self.d * gamma((self.d - 1)/2 + 1)/ gamma(self.d/2 + 1)
        self.threshold = exp(self.epsilon)/(exp(self.epsilon)+1)

    def perturb(self):
        noise_covariance = np.random.normal(0, self.sigma, size=(self.d,self.d))
        for i in range(len(noise_covariance)):
            for j in range(i):
                noise_covariance[i][j] = noise_covariance[j][i]

        noise_reward1 = np.random.normal(0, self.sigma, size=(self.d,1))
        noise_reward2 = np.random.normal(0, self.sigma, size=(self.d,1))
        return noise_covariance, noise_reward1, noise_reward2

    def pack(self, select_context, reward, server):
        covariance = select_context.dot(select_context.T)
        yx = select_context.T.dot(server.theta_h)*select_context
        gradient = (server.transform(np.clip(select_context.T.dot(server.theta_h), -5, 5)) - reward)*select_context
        noise_covariance, noise_reward1, noise_reward2 = self.perturb()

        noised_package = dict()
        noised_package['xx'] = covariance + noise_covariance
        noised_package['yx'] = yx + noise_reward1
        
        noised_package['gradient'] = gradient + noise_reward2
        return noised_package
    
class LDPCovariateMechanism(Mechanism):
    def __init__(self, params):
        super().__init__(params)
        self.k = params['env']['n_action']
        self.sigma = 6*sqrt(2*log(2.5/self.delta))/self.epsilon
        
    def perturb():
        noise_covariance = np.random.normal(0, self.sigma, size=(self.d, self.d))
        for i in range(len(noise_covariance)):
            for j in range(i):
                noise_covariance[i][j] = noise_covariance[j][i]

        noise_reward = np.random.normal(0, self.sigma, size=(self.d,1))
        return noise_covariance, noise_reward

    def Mechanism_process(self, X, Y):
        noise_covariance = np.random.normal(0, self.sigma, size=(self.d,self.d))
        for i in range(len(noise_covariance)):
            for j in range(i):
                noise_covariance[i][j] = noise_covariance[j][i]

        noise_reward = np.random.normal(0, self.sigma, size=(self.d,1))
        return noise_covariance, noise_reward
    
    def pack(self, select_index, select_context, reward, server):
        Xs = [np.zeros((self.d, self.d)) for _ in range(self.k)]
        Xs[select_index] = select_context.dot(select_context.T)

        Ys = [np.zeros((self.d,1)) for _ in range(self.k)]
        Ys[select_index] = reward*select_context

        # Add Mechanism
        for i in range(self.k):
            Xs[i], Ys[i] = self.Mechanism_process(Xs[i], Ys[i])
            
        noised_package = dict()
        noised_package['Xs'] = Xs
        noised_package['Ys'] = Ys
        
        noised_package['select_index'] = select_index
        
        return noised_package