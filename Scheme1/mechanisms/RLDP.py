from mechanisms import Mechanism
from math import sqrt, log, exp, pi
import numpy as np
from scipy.special import gamma

r = 1

class ROLSMechanism(Mechanism):
    def __init__(self, params):
        super().__init__(params)
        self.r = r
        self.sigma = (self.r+2)*sqrt(2*log(2.5/self.delta))/self.epsilon

    def perturb(self):
        noise_covariance = np.random.normal(0, self.sigma, size=(self.d,self.d))
        for i in range(len(noise_covariance)):
            for j in range(i):
                noise_covariance[i][j] = noise_covariance[j][i]

        noise_reward = np.random.normal(0, self.sigma, size=(self.d,1))
        return noise_covariance, noise_reward

    def pack(self, select_context, reward, server):
        covariance = select_context.dot(select_context.T)
        reward_ = reward*select_context

        noise_covariance, noise_reward = self.perturb()

        noised_package = dict()
        noised_package['xx'] = covariance + noise_covariance
        noised_package['yx'] = reward_ + noise_reward
        return noised_package

class RSGDLDPMechanism(Mechanism):
    def __init__(self, params):
        super().__init__(params)
        self.k = params['env']['n_action']
        self.d = params['env']['dimension']
        self.env_s = params['env']['instance_variance']
        self.r = r
        self.B = self.r*(exp(self.epsilon) + 1)/(exp(self.epsilon) - 1) * sqrt(pi) / 2 * self.d * gamma((self.d - 1)/2 + 1)/ gamma(self.d/2 + 1)
        self.threshold = exp(self.epsilon)/(exp(self.epsilon)+1)

        self.sigma = (self.r+2)*sqrt(2*log(2.5/self.delta))/self.epsilon

    def pack(self, select_context, reward, server):
        gradient = (server.transform(select_context.T.dot(server.theta)) - reward)*select_context
        noised_package = dict()
        x = gradient
        x = self.r*x if np.random.uniform()> (1/2+sqrt(x.T.dot(x))/(2*self.r)) else -self.r*x/sqrt(x.T.dot(x))
        prob = np.random.uniform()
        while True:
            z = np.random.normal(0, self.env_s, (self.d, 1))
            z = z/sqrt(z.T.dot(z))*self.B
            if (((prob>self.threshold) and (z.T.dot(x)>0)) or ((prob<=self.threshold) and (z.T.dot(x)<=0))):
                break

        noised_package['gradient'] = z
        noised_package['f_diff'] = server.transform(select_context.T.dot(server.theta)) - reward
        return noised_package

    
class RGLMMechanism(Mechanism):
    def __init__(self, params):
        super().__init__(params)
        self.env_s = params['env']['instance_variance']
        self.r = r
        self.B = self.r*(exp(self.epsilon) + 1)/(exp(self.epsilon) - 1) * sqrt(pi) / 2 * self.d * gamma((self.d - 1)/2 + 1)/ gamma(self.d/2 + 1)
        self.sigma = (self.r+2)*sqrt(2*log(2.5/self.delta))/self.epsilon
        self.threshold = exp(self.epsilon)/(exp(self.epsilon)+1)

    def perturb(self):
        noise_covariance = np.random.normal(0, self.sigma, size=(self.d,self.d))
        for i in range(len(noise_covariance)):
            for j in range(i):
                noise_covariance[i][j] = noise_covariance[j][i]

        noise_reward = np.random.normal(0, self.sigma, size=(self.d,1))
        noise_gradient = np.random.normal(0, self.r * self.sigma, size=(self.d,1))
        return noise_covariance, noise_reward, noise_gradient

    def pack(self, select_context, reward, server):
        covariance = select_context.dot(select_context.T)
        reward_ = select_context.T.dot(server.theta_h)*select_context
        gradient = (server.transform(np.clip(select_context.T.dot(server.theta_h), -5, 5)) - reward)*select_context

        noise_covariance, noise_reward, noise_gradient = self.perturb()

        noised_package = dict()
        noised_package['xx'] = covariance + noise_covariance
        noised_package['yx'] = reward_ + noise_reward
        noised_package['gradient'] = gradient + noise_gradient
        noised_package['f_diff'] = (server.transform(np.clip(select_context.T.dot(server.theta_h), -5, 5)) - reward)
        return noised_package
