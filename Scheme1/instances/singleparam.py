import os
import sys
import glob 
import argparse
import numpy as np
import pandas as pd
from itertools import product
from functools import partial
from multiprocessing import Pool, cpu_count

pdir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(pdir)

from plotters import plotters
from utils import wrap_params, settings_dict

def run(params_func, random_seed, prefix, eps):
    params = params_func(random_seed = random_seed, eps = eps)
    dist_file = prefix + params['name'] + '|'+ 'eps=' + str(eps) + '|' + 'random_seed=' + str(random_seed) + '|.csv'
    worker = settings_dict['worker'][params['worker']](params)
    worker.run()
    worker.save(dist_file)
    return None

def generate_params(args):
    T, d, k, n_random_seeds, prefix, eps = args.time_horizon, args.dimension, args.n_actions, args.n_random_seeds, args.dest, args.eps
    random_seeds = range(n_random_seeds)
    
    common_params = partial(wrap_params, \
                            dimension = d, n_action = k, T = T, \
                            env = 'GaussianLinear', worker = 'A', \
                            reward = 'normal',  context_type = 'gapless', 
                            profiler = 'Profiler', echo = False, echo_freq = 100)

    ldpucb_params = partial(common_params, name = 'LDP-UCB', \
                mechanism = 'OLS', server = 'LUCB', \
                batch = False)

    ldpucbg_params = partial(common_params, name = 'LDP-GLOC',  \
                mechanism = 'GLM', server = 'LUCBG')

    ldpols_params = partial(common_params, name = 'LDP-OLS', \
                mechanism = 'OLS', server = 'GOLS', \
                batch = False)

    ldpsgd_params = partial(common_params, name = 'LDP-SGD',  \
                mechanism = 'SGD', server = 'GSGD', \
                batch = False)

    return [ldpucb_params, ldpucbg_params, ldpols_params, ldpsgd_params], random_seeds, prefix, eps

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--time_horizon', type = int, default = int(1e6),
                        help='total time horizon to run')
    parser.add_argument('--n_random_seeds', type = int, default = 10,
                        help='total amount of random seeds')
    parser.add_argument('--dimension', type = int, default = 2,
                        help='dimension of contexts')
    parser.add_argument('--n_actions', type = int, default = 10,
                        help='number of action')
    parser.add_argument('--eps', type = float, nargs = '+', default=[1, 5],
                        help='privacy epsilon')
    parser.add_argument('--dest', type = str, default = pdir + '/results/single-param/',
                        help='destination folders')
 
    args = parser.parse_args()
    params_funcs, random_seeds, prefix, eps = generate_params(args)
    prefix = [pdir + args.dest]
    if not os.path.exists(prefix[0]):
        os.makedirs(prefix[0])

    n_process = min(len(params_funcs)*len(random_seeds), int(cpu_count()/4))
    print(f'using {n_process} processes')
    with Pool(processes = n_process) as pool:
        collection_source = pool.starmap(run, 
                                        product(params_funcs, random_seeds, prefix, eps))
    plotters.single_exp_plot(prefix[0])
