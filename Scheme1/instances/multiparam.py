import os
import sys
import glob 
import tqdm 
import warnings
import argparse
import numpy as np
import pandas as pd
from itertools import product
from functools import partial
from math import sqrt, exp, pi, gamma, log
from multiprocessing import Pool, cpu_count

pdir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(pdir)

from plotters import plotters
from utils import wrap_params, settings_dict

def run(random_seed, 
        prefix, 
        epsilon, 
        n_action,
        time_horizon,
        dimension = 20,
        echo_freq = 100
        ):
    np.random.seed(random_seed)

    T = time_horizon
    k = n_action
    d = dimension
    s = int(d/k)
    v = 1
    echo_freq = echo_freq

    delta = 0.1
    alpha = 0.1
    sigma = 6*sqrt(2*log(2.5/delta))/epsilon
    warm_up = int(3e3)
    I = np.eye(d)
    B = (exp(epsilon) + 1)/(exp(epsilon) - 1) * sqrt(pi) / 2 * d * gamma((d - 1)/2 + 1)/ gamma(d/2 + 1)
    threshold = exp(epsilon)/(exp(epsilon)+1)
    zeta = 1/sqrt(T)

    OLS_Vs = [np.zeros((d,d)) for _ in range(k)]
    OLS_Rs = [np.zeros((d,1)) for _ in range(k)]

    SGD_Vs = [np.zeros((d,d)) for _ in range(k)]
    SGD_Rs = [np.zeros((d,1)) for _ in range(k)]

    UCB_I = np.eye(d*k)
    UCB_Vs = np.zeros((d*k, d*k))
    UCB_Rs = np.zeros((d*k, 1))

    GLOC_I = np.eye(d*k)
    GLOC_Vs = np.zeros((d*k, d*k))
    GLOC_Rs = np.zeros((d*k, 1))

    def perturb(X, Y):
        noise_covariance = np.random.normal(0, sigma, size=X.shape)
        for i in range(len(noise_covariance)):
            for j in range(i):
                noise_covariance[i][j] = noise_covariance[j][i]

        noise_reward = np.random.normal(0, sigma, size=Y.shape)
        return noise_covariance, noise_reward

    def privacy_process(X, Y):
        noise_covariance, noise_reward = perturb(X, Y)

        noised_package = dict()
        X = X + noise_covariance
        Y = Y + noise_reward
        return X, Y

    # estimator
    theta_sgd = np.random.normal(0, 1, (d, k))
    for i in range(k):
        theta_sgd[:, i] /= sqrt(theta_sgd[:, i].T.dot(theta_sgd[:, i]))

    theta_ols = np.random.normal(0, 1, (d, k))
    for i in range(k):
        theta_ols[:, i] /= sqrt(theta_ols[:, i].T.dot(theta_ols[:, i]))

    theta_ucb = np.random.normal(0, 1, (d*k, 1))
    theta_ucb /= sqrt(theta_ucb.T.dot(theta_ucb))

    theta_gloc = np.random.normal(0, 1, (d*k, 1))
    theta_gloc /= sqrt(theta_gloc.T.dot(theta_gloc))

    theta_gloc_h = np.random.normal(0, 1, (d*k, 1))
    theta_gloc_h /= sqrt(theta_gloc_h.T.dot(theta_gloc_h))

    # theta setting
    # theta = np.zeros((d, k)) 
    # for i in range(k):
    #     theta[i*s:(i+1)*s, i] = 1
    #     theta[:, i] = np.random.normal(0, sqrt(v)/3, (d))
    #     theta[:, i] /= sqrt(theta[:, i].T.dot(theta[:, i]))

    # theta setting 2
    theta = np.random.normal(0, 1, (d, k)) 
    for i in range(k):
        theta[:, i] /= sqrt(theta[:, i].T.dot(theta[:, i]))

    context = np.random.normal(0, 1, (d, 1))
    # context /= sqrt(context.T.dot(context))

    # iterator = tqdm.tqdm(range(T+1))
    iterator = range(T+1)

    times = list()
    est_errors_sgd = list()
    est_errors_ucb = list()
    est_errors_ols = list()
    est_errors_gloc = list()
    pseudo_regrets_sgd = list()
    pseudo_regrets_ucb = list()
    pseudo_regrets_ols = list()
    pseudo_regrets_gloc = list()
    pseudo_cum_regrets_sgd = list()
    pseudo_cum_regrets_ucb = list()
    pseudo_cum_regrets_ols = list()
    pseudo_cum_regrets_gloc = list()

    for t in iterator:
        record = dict()
        # For MAB contextual decision
        select_index_sgd = np.argmax([theta_sgd[:, i].T.dot(context) for i in range(k)])
        select_index_ols = np.argmax([theta_ols[:, i].T.dot(context) for i in range(k)])
        if t<warm_up:
            select_index_sgd = t%k
            select_index_ols = t%k

        # For linear contextual decision
        upsilon_t1 = sigma*sqrt(t)*(4*sqrt(d) + 2*log(2*T/alpha)) # t-1
        upsilon_t = sigma*sqrt(t+1)*(4*sqrt(d) + 2*log(2*T/alpha)) # t-1
        c_t1 = 2*upsilon_t1
        c_t = 2*upsilon_t
        beta = 2*sigma*sqrt(d*log(T)) + (sqrt(3*upsilon_t) + sigma*sqrt(d*(t+1)/upsilon_t))*d*log(T)
        linear_contexts = np.zeros((d*k, k))
        # ucb
        for i in range(k):
            linear_contexts[i*d:(i+1)*d, i] = context.ravel()
        try:
            temp_matrix = np.linalg.inv(UCB_Vs + c_t1 * UCB_I) 
        except:
            temp_matrix = np.identity(d*k)
        action_values = []
        for i in range(linear_contexts.shape[1]):
            x = linear_contexts[:,i]
            UCB_value = theta_ucb.T.dot(x)[0]+beta*sqrt(x.T.dot(temp_matrix).dot(x))
            action_values.append(UCB_value)
        select_index_ucb = np.argmax(action_values)

        # gloc
        try:
            temp_matrix = np.linalg.inv(GLOC_Vs + c_t1 * GLOC_I) # V_{t-1}, c_{t-1}
        except:
            temp_matrix = np.identity(d*k)
        action_values = []
        for i in range(linear_contexts.shape[1]):
            x = linear_contexts[:,i]
            GLOC_value = theta_gloc.T.dot(x)[0]+beta*sqrt(x.T.dot(temp_matrix).dot(x))
            action_values.append(GLOC_value)
        select_index_gloc = np.argmax(action_values)

        # return regret for sgd
        expected_reward_sgd = context.T.dot(theta[:, select_index_sgd])
        noise = np.random.normal(0, 0.05, size=(1,1))
        reward_sgd = expected_reward_sgd + noise
        optimal_value = np.max([context.T.dot(theta[:, i]) for i in range(k)])
        pseudo_regret_sgd = optimal_value - expected_reward_sgd

        # return regret for ols
        expected_reward_ols = context.T.dot(theta[:, select_index_ols])
        noise = np.random.normal(0, 0.05, size=(1,1))
        reward_ols = expected_reward_ols + noise
        optimal_value = np.max([context.T.dot(theta[:, i]) for i in range(k)])
        pseudo_regret_ols = optimal_value - expected_reward_ols

        # return regret for UCB
        expected_reward_ucb = context.T.dot(theta[:, select_index_ucb])
        noise = np.random.normal(0, 0.05, size=(1,1))
        reward_ucb = expected_reward_ucb + noise
        optimal_value = np.max([context.T.dot(theta[:, i]) for i in range(k)])
        pseudo_regret_ucb = optimal_value - expected_reward_ucb

        # return regret for GLOC
        expected_reward_gloc = context.T.dot(theta[:, select_index_gloc])
        noise = np.random.normal(0, 0.05, size=(1,1))
        reward_gloc = expected_reward_gloc + noise
        optimal_value = np.max([context.T.dot(theta[:, i]) for i in range(k)])
        pseudo_regret_gloc = optimal_value - expected_reward_gloc

        # SGD update
        if t<warm_up:  
            SGD_Xs = [np.zeros((d,d)) for _ in range(k)]
            SGD_Xs[select_index_sgd] = context.dot(context.T)

            SGD_Ys = [np.zeros((d,1)) for _ in range(k)]
            SGD_Ys[select_index_sgd] = reward_sgd*context

            i =  select_index_sgd
            SGD_Xs[i], SGD_Ys[i] = privacy_process(SGD_Xs[i], SGD_Ys[i])
            SGD_Vs[i] += SGD_Xs[i] 
            SGD_Rs[i] += SGD_Ys[i]
            theta_sgd[:, i] = np.linalg.inv(SGD_Vs[i] + c_t * I).dot(SGD_Rs[i]).ravel()
        else:
            gradients = np.zeros((d*k))
            gradients[(select_index_sgd*d): ((select_index_sgd+1)*d)] = ((context.T.dot(theta_sgd[:, select_index_sgd]) - reward_sgd)*context).ravel()

            x = gradients
            if (x.T.dot(x)>0):
                x = x/sqrt(x.T.dot(x)) if np.random.uniform()> (1/2+sqrt(x.T.dot(x))/(2)) else -x/sqrt(x.T.dot(x))
            else:
                raise Exception("Sorry, x is a zero vector")
            prob = np.random.uniform()
            while True:
                z = np.random.normal(0, 1, (d*k, 1))
                z = z/sqrt(z.T.dot(z))*B
                if (((prob>threshold) and (z.T.dot(x)>0)) or ((prob<=threshold) and (z.T.dot(x)<=0))):
                    break
            eta = k*5/(t+1)
            for i in range(k):
                theta_sgd[:, i] = theta_sgd[:, i] - eta*(z[(i*d): ((i+1)*d)].ravel() + 0.05*theta_sgd[:, i]) 

        # OLS update
        OLS_Xs = [np.zeros((d,d)) for _ in range(k)]
        OLS_Xs[select_index_ols] = context.dot(context.T)

        OLS_Ys = [np.zeros((d,1)) for _ in range(k)]
        OLS_Ys[select_index_ols] = reward_ols*context

        if t<warm_up:  
            i = select_index_ols
            OLS_Xs[i], OLS_Ys[i] = privacy_process(OLS_Xs[i], OLS_Ys[i])
            OLS_Vs[i] += OLS_Xs[i] 
            OLS_Rs[i] += OLS_Ys[i]
            theta_ols[:, i] = np.linalg.inv(OLS_Vs[i] + c_t * I).dot(OLS_Rs[i]).ravel()
        else:
            for i in range(k):
                OLS_Xs[i], OLS_Ys[i] = privacy_process(OLS_Xs[i], OLS_Ys[i])
                OLS_Vs[i] += OLS_Xs[i] 
                OLS_Rs[i] += OLS_Ys[i]
                theta_ols[:, i] = np.linalg.inv(OLS_Vs[i] + c_t * I).dot(OLS_Rs[i]).ravel()

        # UCB Update
        X, Y = privacy_process(linear_contexts[:, select_index_ucb].reshape((-1, 1)).dot(linear_contexts[:, select_index_ucb].reshape((-1, 1)).T),\
                               reward_ucb*linear_contexts[:, select_index_ucb].reshape((-1, 1)))
        UCB_Vs += X
        UCB_Rs += Y
        theta_ucb = np.linalg.inv(UCB_Vs + c_t * UCB_I).dot(UCB_Rs)

        # GLOC Update
        linear_context =  linear_contexts[:, [select_index_gloc]]
        X, Y = privacy_process(linear_context.T.dot(linear_context),\
                               linear_context.T.dot(theta_gloc_h)*linear_contexts[:, [select_index_gloc]])
        GLOC_Vs += X
        GLOC_Rs += Y
        theta_gloc = np.linalg.inv(GLOC_Vs + c_t * GLOC_I).dot(GLOC_Rs)

        nabla_h = (theta_gloc_h.T.dot(linear_context) - reward_gloc)*linear_context 
        nabla_h += np.random.normal(0, 2*sigma, size=nabla_h.shape)
        theta_gloc_h -= zeta*nabla_h
        
        # Update next contex
        context = np.random.normal(0, 1, (d, 1))

        # Records
        pseudo_regrets_sgd.append(pseudo_regret_sgd[0])
        pseudo_regrets_ols.append(pseudo_regret_ols[0])
        pseudo_regrets_ucb.append(pseudo_regret_ucb[0])
        pseudo_regrets_gloc.append(pseudo_regret_gloc[0])

        if (t%echo_freq==0):
            times.append(t)
            est_errors_sgd.append(sum([np.linalg.norm(theta_sgd[:, [i]] - theta[:, [i]]) for i in range(k)]))
            est_errors_ols.append(sum([np.linalg.norm(theta_ols[:, [i]] - theta[:, [i]]) for i in range(k)]))
            est_errors_ucb.append(sum([np.linalg.norm(theta_ucb[i*d:(i+1)*d, :] - theta[:, [i]]) for i in range(k)]))
            est_errors_gloc.append(sum([np.linalg.norm(theta_gloc[i*d:(i+1)*d, :] - theta[:, [i]]) for i in range(k)]))
            pseudo_cum_regrets_sgd.append(np.sum(pseudo_regrets_sgd))
            pseudo_cum_regrets_ols.append(np.sum(pseudo_regrets_ols))
            pseudo_cum_regrets_ucb.append(np.sum(pseudo_regrets_ucb))
            pseudo_cum_regrets_gloc.append(np.sum(pseudo_regrets_gloc))

    df = pd.DataFrame({'time':times, 'sgdest': est_errors_sgd, 'ucbest': est_errors_ucb, 'glocest': est_errors_gloc, \
                       'olsest': est_errors_ols, 'sgdr': pseudo_cum_regrets_sgd, 'ucbr': pseudo_cum_regrets_ucb, \
                       'olsr': pseudo_cum_regrets_ols,
                       'glocr': pseudo_cum_regrets_gloc})

    df.to_csv(prefix + 'multi_param' + '|'+ 'eps=' + str(epsilon) +  '|'+ 'n_action=' + str(k) + '|' + 'random_seed=' + str(random_seed) + '|.csv')
    return None
    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--time_horizon', type = int, default = int(1e6),
                        help='total time horizon to run')
    parser.add_argument('--n_random_seeds', type = int, default = 10,
                        help='total amount of random seeds')
    parser.add_argument('--n_actions', type = int, default = 3,
                        help='number of action')
    parser.add_argument('--dimension', type = int, default = 2,
                        help='dimension of context')
    parser.add_argument('--eps', type = float, nargs = '+', default=[1, 5],
                        help='privacy epsilon')
    parser.add_argument('--echo_freq', type = int, default=1000,
                        help='echo frequency')
    parser.add_argument('--dest', type = str, default = pdir + '/results/multi-param/',
                        help='destination folders')
 
    args = parser.parse_args()
    prefix = [pdir + args.dest]
    random_seeds = list(range(args.n_random_seeds))
    if not os.path.exists(prefix[0]):
        os.makedirs(prefix[0])

    n_process = min(len(args.eps)*args.n_random_seeds, 3)
    print(f'using {n_process} processes')
    with Pool(processes = n_process) as pool:
        collection_source = pool.starmap(run, 
                                        product(random_seeds, 
                                                prefix, 
                                                args.eps,
                                                [args.n_actions],
                                                [args.time_horizon]))

    plotters.multiparam_exp_plot(prefix[0], 'Multiparam')

