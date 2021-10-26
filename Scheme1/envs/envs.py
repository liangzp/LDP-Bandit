from envs import Environment
import numpy as np
from math import log, sqrt, exp
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, MinMaxScaler

class GaussianLinearEnvironment(Environment):
    def __init__(self, params):
        super().__init__(params)
        self.theta = np.random.normal(0, self.env_s, size=(self.d, 1))
        self.theta = self.theta/sqrt(self.theta.T.dot(self.theta))
        self.context_type = params['env']['context_type']
        self.mode = params['env']['yx']
        if self.mode == 'normal':
            self.transform = lambda x: x
            self.rg = lambda x: x
        elif self.mode == 'poisson':
            self.transform = lambda x: exp(x)
            self.rg = lambda x: np.random.poisson(x)
        elif self.mode == 'logistic':
            self.transform = lambda x: 1/(1 + exp(-x))
            self.rg = lambda x: np.random.binomial(1, x, 1)[0]
    
    def step(self, select_index):    
        select_context = self.cache_contexts[:, select_index].reshape((-1,1))
        expected_reward = np.max(self.transform(self.theta.T.dot(select_context)))
        
        optimal_value = list()
        for i in range(self.cache_contexts.shape[1]):
            optimal_value.append(self.transform(self.theta.T.dot(self.cache_contexts[:, i])))
        optimal_value = np.max(optimal_value)
        pseudo_regret = optimal_value - expected_reward
        reward = self.rg(expected_reward)
        next_contexts = self.generate_contexts()
        return reward, pseudo_regret, next_contexts
    
    def initialize(self):
        init_contexts = self.generate_contexts()
        return init_contexts
    
    def generate_contexts(self):
        contexts = np.random.normal(0, self.env_s, (self.d, self.k))
        for i in range(contexts.shape[1]):
            contexts[:, i] /= sqrt(contexts[:, i].T.dot(contexts[:, i]))
        self.cache_contexts = contexts
        return np.array(contexts)
    
class LoanEnvironment(Environment):
    def __init__(self, params):
        super().__init__(params)
        self.t = 0
        self.load_data(params['env']['data_id'], params['env']['n_action'])

    def load_data(self, data_id, num_action):
        if data_id == 'online_leaning':
            features_cols = ['apply', 'Primary_FICO', 'onemonth', 'Term', 'Amount_Approved',\
                'CarType', 'Competition_rate'
                ]
                
            df = pd.read_csv('data/CPRM_AutoLOan_OnlineAutoLoanData.csv')
            # impute price to fit logistic regression
            q = 1/(1+df['onemonth']/100)
            p = (df['mp'] * (q * (1 - q**df['Term'])/(1 - q)) - df['Amount_Approved']).values.reshape((-1,1))
            
            df = df[features_cols]
            df['constant'] = 1
            
            # preprocess
            if 'State' in features_cols:
                df['State'] = df['State'].fillna('Unknown')
            if 'Previous_Rate' in features_cols:
                df['Previous_Rate'] = df['Previous_Rate'].fillna(0)
            df = df.infer_objects()
            if 'State' in features_cols:
                df[['State']] = df[['State']].astype("category")
                
            if 'Type' in features_cols:
                df[['Type']] = df[['Type']].astype("category")

            if 'CarType' in features_cols:
                df[['CarType']] = df[['CarType']].astype("category")

            df = pd.get_dummies(df)
            X_, y = df.drop('apply', axis = 1), df['apply']
            X_p = (p* X_).add_suffix('_p')
            X = pd.concat([X_, X_p], axis = 1)
            ss = MinMaxScaler().fit(X)
            X_train = ss.transform(X)     

            clf = LogisticRegression(random_state=0, fit_intercept=False).fit(X_train, y)
            
            indices = np.array(range(X.shape[0]))
            np.random.shuffle(indices)
            self.theta = clf.coef_.reshape((-1, 1))
            self.X = X_.iloc[indices]
            self.scaler = ss
            self.prices = np.linspace(0, -25000, num = num_action)
            self.d = 2*X_.shape[1]
            self.transform = lambda x: 1/(1 + exp(-x))
            self.rg = lambda x: np.random.binomial(1, x, 1)[0]
            self.model = clf
        
    def step(self, select_index):    
        select_context = self.cached_contexts[:, [select_index]]
        expected_reward = self.transform(self.theta.T.dot(select_context))
        optimal_value = np.max([self.transform(self.theta.T.dot(self.cached_contexts[:, [i]])) for i in range(self.k)])
        pseudo_regret = optimal_value - expected_reward
        reward = expected_reward
        next_contexts = self.generate_contexts()
        self.t += 1
        return reward, pseudo_regret, next_contexts
    
    def initialize(self):
        init_contexts = self.generate_contexts()
        return init_contexts
    
    def generate_contexts(self):
        record = self.X.iloc[self.t]
        contexts = np.zeros((self.d, self.k))
        for j, price in enumerate(self.prices):
            contexts[:, [j]] = self.scaler.transform(pd.concat([record, record * price], axis = 0).values.reshape((1,-1))).reshape((-1,1))
        self.cached_contexts = contexts
        return contexts

