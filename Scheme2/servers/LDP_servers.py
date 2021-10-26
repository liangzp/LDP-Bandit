from servers import Server
import numpy as np
from math import sqrt, log, exp

class NPUCBserver(Server):
    def __init__(self,params):
        super().__init__(params)
        self.initialize(params)
        
    def decide(self, contexts):
        try:
            temp_matrix = np.linalg.inv(self.V + self.c * self.I) # can drop this c
        except:
            temp_matrix = np.identity(self.d)
        action_values = []
        for i in range(contexts.shape[1]):
            x = contexts[:,i]
            UCB_value = self.theta.T.dot(x)+self.beta*sqrt(x.T.dot(temp_matrix).dot(x))
            action_values.append(UCB_value)
        select_index = np.argmax(action_values)
        select_context = contexts[:, select_index].reshape((-1,1))
        return select_index, select_context
        
    def initialize(self, params):
        self.alpha = params['fail_prob']
        self.V = np.zeros((self.d, self.d))
        self.u = np.zeros((self.d, 1))
        self.theta = np.zeros((self.d, 1))
        self.t = 1
        self.l = 1
        self.c = 10
        self.update_params()

    def update(self, t, output):
        covariance = output['xx']
        reward = output['yx']
        self.t = t + 1
        self.V += covariance
        self.u += reward
        self.theta = np.linalg.inv(self.V + self.c*self.I).dot(self.u)
        self.update_params()

    def update_params(self):
        self.beta = self.l**(1/2)+sqrt(2*log(1/self.alpha)+self.d*log(1+self.t/(self.l*self.d)))

class LDPUCBServer(Server):
    def __init__(self,params):
        super().__init__(params)
        self.initialize(params)
        
    def initialize(self, params):
        self.alpha = params['fail_prob']
        self.V = np.eye(self.d)
        self.u = np.zeros((self.d, 1))
        self.theta = params['server']['init_theta']
        self.sigma = 6*sqrt(2*log(2.5/self.delta))/self.epsilon
        self.c = 0
        self.t = 1
        self.update_params()
        
    def decide(self, contexts):
        try:
            temp_matrix = np.linalg.inv(self.V + self.c * self.I) 
        except:
            temp_matrix = np.identity(self.d)
        action_values = []
        for i in range(contexts.shape[1]):
            x = contexts[:,i]
            UCB_value = self.theta.T.dot(x)[0]+self.beta*sqrt(x.T.dot(temp_matrix).dot(x))
            action_values.append(UCB_value)
        select_index = np.argmax(action_values)
        select_context = contexts[:, select_index].reshape((-1,1))
        return select_index, select_context
         
    def update(self, t, noise_output):
        covariance = noise_output['xx']
        reward = noise_output['yx']
        self.t = t + 1
        self.V += covariance
        self.u += reward
        self.theta = np.linalg.inv(self.V + self.c*self.I).dot(self.u)
        self.update_params()
        
    def update_params(self):
        self.upsilon = self.sigma*sqrt(self.t)*(4*sqrt(self.d) + 2*log(2*self.T/self.alpha)) 
        self.c = 2*self.upsilon/10
        self.beta = 2*self.sigma*sqrt(self.d*log(self.T)) + (sqrt(3*self.upsilon) + self.sigma*sqrt(self.d*self.t/self.upsilon))*self.d*log(self.T)

class GOLSServer(Server):
    def __init__(self,params):
        super().__init__(params)
        self.initialize(params)
         
    def update(self,  t, noise_output):
        covariance = noise_output['xx']
        reward = noise_output['yx']
        self.V += covariance
        self.u += reward
        
        self.t = t + 1
        self.upsilon = self.sigma*sqrt(self.t)*(4*sqrt(self.d) + 2*log(2*self.T/self.alpha)) 
        self.c = 2*self.upsilon
        self.theta = np.linalg.inv(self.V + self.c*self.I).dot(self.u)
        
    def initialize(self, params):
        self.alpha = params['fail_prob']
        self.V = np.zeros((self.d, self.d))
        self.u = np.zeros((self.d, 1))
        self.c = 0
        self.t = 1
        self.sigma = 6*sqrt(2*log(2.5/self.delta))/self.epsilon
        self.theta = params['server']['init_theta']
        self.reward = params['env']['yx']
        if self.reward != 'normal':
            raise Exception('Matrix-based Algorithm can only deal with normal linear model')
            
#         if (params['server']['batch']):
#             T = int(params['env']['time_horizon'])    
#             d = int(params['env']['dimension'])
#             M = int(params['num_batch'])
#             self.grids = self.generate_grid(T, d)
#             self.b = 0
#             self.update = self.update_batch
#         else:
#             self.update = self.update_nobatch
    
    def generate_grid(self, T, d):
        C = 1
        a = d
        t = a
        grids = list()
        grids.append(t)
        while True:
            if ((T-grids[-1])/T < 0.01):
                break
            t = 2*t
            grids.append(t)
        if (grids[-1]<T):
            grids.append(T)
        return grids
    
    def decide(self, contexts):
        action_values = []
        for i in range(contexts.shape[1]):
            x = contexts[:,i]
            action_value = self.theta.T.dot(x)
            action_values.append(action_value)
        select_index = np.argmax(action_values)
        select_context = contexts[:, select_index].reshape((-1,1))
        return select_index, select_context

#     def update_nobatch(self, t, outputs):
#         V = outputs['xx']
#         yx = outputs['yx']
#         self.V += V
#         self.u += yx
#         self.theta = np.linalg.inv(self.V).dot(self.u)
        
#     def update_batch(self, t, outputs):
#         V = outputs['xx']
#         yx = outputs['yx']
#         self.V += V
#         self.u += yx
#         if (t>self.grids[self.b]):
#             self.theta = np.linalg.inv(self.V).dot(self.u)
#             self.V = np.zeros((self.d, self.d))
#             self.u = np.zeros((self.d, 1))
#             self.b += 1

class GreedySGDServer(Server):
    def __init__(self,params):
        super().__init__(params)
        self.initialize(params)
          
    def initialize(self, params):
        self.theta = params['server']['init_theta']
        self.K = params['env']['n_action']
        self.d = params['env']['dimension']
#         self.theta = self.theta/sqrt(self.theta.T.dot(self.theta))
        self.reward = params['env']['yx']
        if self.reward == 'normal':
            self.transform = lambda x: x
            self.lambda_ = 0.00
            self.eta_constant = 1000
        elif self.reward == 'poisson':
            self.transform = lambda x: exp(x)
            self.lambda_ = 0.05
            self.eta_constant = 5
        elif self.reward == 'logistic':
            self.transform = lambda x: 1/(1 + exp(-x))
            self.lambda_ = 0.05
            self.eta_constant = 5
        
    def decide(self, contexts):
        action_values = []
        for i in range(contexts.shape[1]):
            x = contexts[:,i]
            action_value = self.theta.T.dot(x)
            action_values.append(action_value)
        select_index = np.argmax(action_values)
        select_context = contexts[:, select_index].reshape((-1,1))
        return select_index, select_context

    def update(self, t, outputs):
        gradient = outputs['gradient']
        eta = (sqrt(self.K)*self.d)/(sqrt(t+1)*self.eta_constant)
        self.theta = self.theta - eta* (gradient + self.lambda_*self.theta)
        
class LDPGLMServer(Server):
    def __init__(self,params):
        super().__init__(params)
        self.initialize(params)
        
    def initialize(self, params):
        self.alpha = params['fail_prob']
        self.V_t = np.zeros((self.d, self.d))
        self.V_h = np.zeros((self.d, self.d))
        self.u = np.zeros((self.d, 1))
        self.I = np.eye(self.d)
        
        self.K = params['env']['n_action']
        self.d = params['env']['dimension']
        
#         self.theta_h = np.random.normal(0, 1, size=(self.d, 1))
#         self.theta_h = self.theta_h/sqrt(self.theta_h.T.dot(self.theta_h))
        
#         self.theta_t = np.random.normal(0, 1, size=(self.d, 1))
#         self.theta_t = self.theta_t/sqrt(self.theta_t.T.dot(self.theta_t))
        
        self.theta_h = params['server']['init_theta']
        self.theta_t = params['server']['init_theta']
        
        self.theta = self.theta_h
        self.sigma = 6*sqrt(2*log(3.75/self.delta))/self.epsilon
        self.zeta = (self.d*self.K)/sqrt(self.T)
        self.reward = params['env']['yx']
        print(self.reward)
        if self.reward == 'normal':
            self.transform = lambda x: x
            self.mu = 1
        elif self.reward == 'poisson':
            self.transform = lambda x: exp(x)
            self.mu = exp(-1)
        elif self.reward == 'logistic':
            self.transform = lambda x: 1/(1 + exp(-x))
            self.mu = exp(-1)/(exp(-1)+1)**2
        self.c = 0
        self.t = 1
        self.update_params()
        
    def decide(self, contexts):
        try:
            temp_matrix = np.linalg.inv(self.V + self.c * self.I) 
        except:
            temp_matrix = np.identity(self.d)
        action_values = []
        for i in range(contexts.shape[1]):
            x = contexts[:,i]
            UCB_value = self.theta_t.T.dot(x)+self.beta*sqrt(x.T.dot(temp_matrix).dot(x))
            action_values.append(UCB_value)
        select_index = np.argmax(action_values)
        select_context = contexts[:, select_index].reshape((-1,1))
        return select_index, select_context
         
    def update(self, t, noise_output):
        covariance = noise_output['xx']
        reward = noise_output['yx']
        gradient = noise_output['gradient']
        self.t = t + 1
        self.V_t += covariance
        self.u += reward
        self.theta_t = np.linalg.inv(self.V_t + self.c*self.I).dot(self.u)
        self.update_params()
        self.V_h = self.V_t + self.c*self.I
        self.theta_h = self.theta_h - self.zeta*gradient
        self.theta = self.theta_h
        
    def update_params(self):
        self.upsilon = self.sigma*sqrt(self.t)*(4*sqrt(self.d) + 2*log(2*self.T/self.alpha)) 
        self.c = 2*self.upsilon
        self.beta = sqrt(self.sigma/self.mu*sqrt(self.d*self.t))
        
class RLUCBServer(Server):
    def __init__(self,params):
        super().__init__(params)
        self.initialize(params)
        
    def initialize(self, params):
        self.alpha = params['fail_prob']
        self.V = np.zeros((self.d, self.d))
        self.u = np.zeros((self.d, 1))
        self.theta = np.zeros((self.d, 1))
        self.r = 385
        self.sigma = (self.r + 4)*sqrt(2*log(2.5/self.delta))/self.epsilon
        self.c = 0
        self.t = 1
        self.update_params()
        
    def decide(self, contexts):
        try:
            temp_matrix = np.linalg.inv(self.V + self.c * self.I)
        except:
            temp_matrix = np.identity(self.d)
        action_values = []
        for i in range(contexts.shape[1]):
            x = contexts[:,i]
            UCB_value = self.theta.T.dot(x)+self.beta*sqrt(x.T.dot(temp_matrix).dot(x))
            action_values.append(UCB_value)
        select_index = np.argmax(action_values)
        select_context = contexts[:, select_index].reshape((-1,1))
        return select_index, select_context
         
    def update(self, t, noise_output):
        covariance = noise_output['xx']
        reward = noise_output['yx']
        self.t = t + 1
        self.V += covariance
        self.u += reward
        self.theta = np.linalg.inv(self.V + self.c*self.I).dot(self.u)
        self.update_params()
        
    def update_params(self):
        self.upsilon = self.sigma*sqrt(self.t)*(4*sqrt(self.d) + 2*log(2*self.T/self.alpha)) 
        self.c = 2*self.upsilon
        self.beta = 2*self.sigma*sqrt(self.d*log(self.T)) + (sqrt(3*self.upsilon) + self.sigma*sqrt(self.d*self.t/self.upsilon))*self.d*log(self.T)
        
class ROLSServer(Server):
    def __init__(self,params):
        super().__init__(params)
        self.initialize(params)
 
    def initialize(self, params):
        self.alpha = params['fail_prob']
        self.V = np.zeros((self.d, self.d))
        self.u = np.zeros((self.d, 1))
        self.theta = np.zeros((self.d, 1))
        self.reward = params['env']['yx']
        self.sigma = 6*sqrt(2*log(2.5/self.delta))/self.epsilon
        self.c = 0
        self.t = 1
        self.update_params()
        if self.reward != 'normal':
            raise Exception('Matrix-based Algorithm can only deal with normal linear model')
            
        if (params['server']['batch']):
            T = int(params['env']['time_horizon'])    
            d = int(params['env']['dimension'])
            M = int(params['num_batch'])
            self.grids = self.generate_grid(T, d, M)
            self.b = 0
            self.update = self.update_batch
        else:
            self.update = self.update_nobatch
        
    def generate_grid(self, T, d, M):
        C = 1
        a = d
        t = a
        grids = list()
        grids.append(t)
        t = 5 * d**2*(sqrt(d) + log(T))**(2) + d
        grids.append(t)
        while True:
            if ((T-grids[-1])/T < 0.02):
                break
            t = 3*grids[-1] - 2*grids[-2]
            grids.append(t)
        if (grids[-1]<T):
            grids.append(T)
        return grids
    
    def decide(self, contexts):
        action_values = []
        for i in range(contexts.shape[1]):
            x = contexts[:,i]
            action_value = self.theta.T.dot(x)
            action_values.append(action_value)
        select_index = np.argmax(action_values)
        select_context = contexts[:, select_index].reshape((-1,1))
        return select_index, select_context

    def update_nobatch(self, t, outputs):
        xx = outputs['xx']
        yx = outputs['yx']
        self.V += xx
        self.u += yx
        self.t = t + 1
        self.theta = np.linalg.inv(self.V + self.c*self.I).dot(self.u)
        self.update_params()
        
    def update_batch(self, t, outputs):
        xx = outputs['xx']
        yx = outputs['yx']
        self.V += xx
        self.u += yx
        self.t = t + 1
        self.update_params()
        if (t>self.grids[self.b]):
            self.theta = np.linalg.inv(self.V + self.c*self.I).dot(self.u)
            self.V = np.zeros((self.d, self.d))
            self.u = np.zeros((self.d, 1))
            self.b += 1
        
    def update_params(self):
        self.upsilon = self.sigma*sqrt(self.t)*(4*sqrt(self.d) + 2*log(2*self.T/self.alpha)) 
        self.c = 2*self.upsilon
        self.beta = 2*self.sigma*sqrt(self.d*log(self.T)) + (sqrt(3*self.upsilon) + self.sigma*sqrt(self.d*self.t/self.upsilon))*self.d*log(self.T)
        
class RSGDServer(Server):
    def __init__(self,params):
        super().__init__(params)
        self.initialize(params)
          
    def initialize(self, params):
        self.theta = np.random.normal(0, 18, size=(self.d, 1))
        self.theta = self.theta/sqrt(self.theta.T.dot(self.theta))
        self.reward = params['env']['yx']
        if self.reward == 'normal':
            self.transform = lambda x: x
        elif self.reward == 'poisson':
            self.transform = lambda x: exp(x)
        elif self.reward == 'logistic':
            self.transform = lambda x: 1/(1 + exp(-x))
    
    def set_param(self, theta):
        self.theta = theta
        
    def decide(self, contexts):
        action_values = []
        for i in range(contexts.shape[1]):
            x = contexts[:,i]
            action_value = self.theta.T.dot(x)
            action_values.append(action_value)
        select_index = np.argmax(action_values)
        select_context = contexts[:, select_index].reshape((-1,1))
        return select_index, select_context

    def update(self, t, outputs):
        gradient = outputs['gradient']
        eta = 1/(t+50)
        self.theta = self.theta - eta*gradient

class RGLMServer(Server):
    def __init__(self,params):
        super().__init__(params)
        self.initialize(params)
        
    def initialize(self, params):
        self.alpha = params['fail_prob']
        self.V_t = np.zeros((self.d, self.d))
        self.V_h = np.zeros((self.d, self.d))
        self.u = np.zeros((self.d, 1))
        self.I = np.eye(self.d)
                
        self.theta_h = np.random.normal(0, 18, size=(self.d, 1))
        self.theta_h = self.theta_h/sqrt(self.theta_h.T.dot(self.theta_h))
        
        self.theta_t = np.random.normal(0, 18, size=(self.d, 1))
        self.theta_t = self.theta_t/sqrt(self.theta_t.T.dot(self.theta_t))
        
        self.theta = self.theta_h
        self.sigma = 6*sqrt(2*log(3.75/self.delta))/self.epsilon
        self.zeta = 1/sqrt(self.T)
        self.reward = params['env']['yx']
        if self.reward == 'normal':
            self.transform = lambda x: x
            self.mu = 1
        elif self.reward == 'poisson':
            self.transform = lambda x: exp(x)
            self.mu = exp(-1)
        elif self.reward == 'logistic':
            self.transform = lambda x: 1/(1 + exp(-x))
            self.mu = exp(-1)/(exp(-1)+1)**2
        self.c = 0
        self.t = 1
        self.update_params()

    def set_param(self, theta):
        self.theta = theta
        
    def decide(self, contexts):
        try:
            temp_matrix = np.linalg.inv(self.V + self.c * self.I) 
        except:
            temp_matrix = np.identity(self.d)
        action_values = []
        for i in range(contexts.shape[1]):
            x = contexts[:,i]
            UCB_value = self.theta_t.T.dot(x)+self.beta*sqrt(x.T.dot(temp_matrix).dot(x))
            action_values.append(UCB_value)
        select_index = np.argmax(action_values)
        select_context = contexts[:, select_index].reshape((-1,1))
        return select_index, select_context
         
    def update(self, t, noise_output):
        covariance = noise_output['xx']
        reward = noise_output['yx']
        gradient = noise_output['gradient']
        self.t = t + 1
        self.V_t += covariance
        self.u += reward
        self.theta_t = np.linalg.inv(self.V_t + self.c*self.I).dot(self.u)
        self.update_params()
        self.V_h = self.V_t + self.c*self.I
        self.theta_h = self.theta_h - self.zeta*gradient
        self.theta = self.theta_h
        
    def update_params(self):
        self.upsilon = self.sigma*sqrt(self.t)*(4*sqrt(self.d) + 2*log(2*self.T/self.alpha)) 
        self.c = 2*self.upsilon
        self.beta = sqrt(self.sigma/self.mu*sqrt(self.d*self.t))
        
class LDPCovariateServer(Server):
    def __init__(self, params):
        super().__init__(params)
        self.initialize(params)
        
    def initialize(self, params):
        self.k = params['env']['n_action']
        self.Vs = [np.zeros((self.d, self.d)) for _ in range(self.k)]
        self.Rs = [np.zeros((self.d,1)) for _ in range(self.k)]
        self.theta = np.random.normal(0, 1, (self.d, self.k))
        for i in range(self.k):
            self.theta[:, i] /= sqrt(self.theta[:, i].T.dot(self.theta[:, i]))
        
    def decide(self, t, context):
        select_index = np.argmax([self.theta[:, i].T.dot(context) for i in range(self.k)])
        if t<(self.k*self.d)**2:
            select_index = t%self.k
        return select_index, context
            
    def update(self, t, outputs):
        for i in range(self.k):
            self.Vs[i] += outputs['Xs'][i]
            self.Rs[i] += outputs['Ys'][i]

        if outputs['select_index'] == -1:
            for i in range(self.k):
                self.theta[:, i] = np.linalg.inv(self.Vs[i]).dot(self.Rs[i]).ravel()



        
    
