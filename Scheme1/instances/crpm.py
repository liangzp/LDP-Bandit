import os
import sys
import glob 
import warnings
import argparse
import numpy as np
import pandas as pd
from multiprocessing import Pool, cpu_count
from itertools import product
from functools import partial

warnings.filterwarnings('ignore')

pdir = os.path.abspath(os.path.dirname(os.path.dirname(__file__)))
sys.path.append(pdir)

from utils import wrap_params, settings_dict
from plotters import plotters

def run(params_func, random_seed, prefix, eps):
    params = params_func(random_seed = random_seed, eps = eps)
    dist_file = prefix + params['name'] + '|'+ 'eps=' + str(eps) + '|' + 'random_seed=' + str(random_seed) + '|.csv'
    worker = settings_dict['worker'][params['worker']](params)
    worker.run()
    worker.save(dist_file)
    return None

def generate_params(args):
    T, k, n_random_seeds, prefix, eps = args.time_horizon, args.n_actions, args.n_random_seeds, args.dest, args.eps
    random_seeds = range(n_random_seeds)
    
    common_params = partial(wrap_params, random_seed = 0, \
                        env = 'OnlineLoan', data_id = 'online_leaning', \
                        dimension = 18, num_batch = int(1e3),\
                        n_action = k, T = T, optimal_gap = 0.1, \
                        context_type = 'gapless', batch = False, \
                        worker = 'B', profiler = 'Profiler', echo = False, echo_freq = 10)

    # normal_common_params_func = partial(common_params, reward = 'normal')

    # ldpucb_params_func = partial(normal_common_params_func, name = 'LDP-UCB', \
    #             mechanism = 'ROLS', server = 'RLUCB', \
    #             )

    # ldpols_params_func = partial(normal_common_params_func, name = 'LDP-OLS',  \
    #             mechanism = 'ROLS', server = 'ROLS', \
    #             )

    logistic_common_params_func = partial(common_params, reward = 'logistic')

    ldbucbglm_params_func = partial(logistic_common_params_func, name = 'LDP-GLOC',  \
                mechanism = 'RGLM', server = 'RGLM', \
                )

    ldpsgd_params_func = partial(logistic_common_params_func, name = 'LDP-SGD',  \
                mechanism = 'RSGD', server = 'RSGD', \
                )
                
    return [ldbucbglm_params_func, ldpsgd_params_func], random_seeds, prefix, eps

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--time_horizon', type = int, default = int(1e5),
                        help='total time horizon to run')
    parser.add_argument('--n_random_seeds', type = int, default = 10,
                        help='total amount of random seeds')
    parser.add_argument('--n_actions', type = int, default = 25,
                        help='number of action')
    parser.add_argument('--eps', type = float, nargs = '+', default=[1],
                        help='privacy epsilon')
    parser.add_argument('--dest', type = str, default = pdir + '/results/crpm/',
                        help='destination folders')
 
    args = parser.parse_args()
    params_funcs, random_seeds, prefix, eps = generate_params(args)
    prefix = [pdir + args.dest]
    if not os.path.exists(prefix[0]):
        os.makedirs(prefix[0])

    n_process = min(len(params_funcs)*len(random_seeds), 6)
    print(f'using {n_process} processes')
    with Pool(processes = n_process) as pool:
        collection_source = pool.starmap(run, 
                                        product(params_funcs, random_seeds, prefix, eps))
    plotters.crpm_exp_plot(prefix[0])
