import glob
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tikzplotlib

sns.set_style("ticks")
plt.rc('font', size=6)
plt.rc('text', usetex=False)
plt.rc('font', family='serif')

def smoother(x, a=0.9, w=10, mode="moving"):
    if mode == "moving":
        y = [x[0]]
        for i in range(1, len(x)):
            y.append((1 - a) * x[i] + a * y[i - 1])
    elif mode == "window":
        y = []
        for i in range(len(x)):
            y.append(np.mean(x[max(i - w, 0):i + 1]))
    return y

def plot_curve(
        ax,
        datas,
        freq=1000,
        label=None,
        feature=None,
        color="black",
        smooth_coef=0.95,
        shaded_err=False,
        shaded_std=True,
        shared_area=0.5,
        **plot_kwargs,
):
    x = datas[0]['time'].tolist()[::freq]
    ys = [np.asarray(data[feature].tolist()[::freq]) for data in datas]
    y_mean = np.mean(ys, axis=0)
    if label is None:
        lin = ax.plot(x, y_mean, color=color, **plot_kwargs)
    else:
        lin = ax.plot(x, y_mean, label=label, color=color, **plot_kwargs)
    if len(ys) > 1:
        y_std = np.std(ys, axis=0) * shared_area
        y_stderr = y_std / np.sqrt(len(ys))
        if shaded_err:
            ax.fill_between(x, y_mean - y_stderr, y_mean + y_stderr, color=color, alpha=.4)
        if shaded_std:
            ax.fill_between(x, y_mean - y_std, y_mean + y_std, color=color, alpha=.2)
    return lin

def parse_dir(repo):
    n = pd.read_csv(list(glob.glob(repo+'/*.csv'))[0]).shape[0]
    settings_dict = dict()
    settings_dict['exp'] = list()
    for data in glob.glob(repo+'/*.csv'):
        settings_dict['exp'].append(data.split('|')[0])
        for item in data.split('|')[1:-1]:
            key, value = item.split('=')
            if not (key in settings_dict):
                settings_dict[key] = list()
            if key == 'reward':
                settings_dict[key].append(value)
            else:
                settings_dict[key].append(int(value))
    for key in settings_dict:
        settings_dict[key] = sorted(list(set(settings_dict[key])))

    return n, settings_dict

def single_exp_plot(repo):
    COLORS = sns.color_palette("tab10")

    label = 'cum_regrets' #'cum_regrets' #'estimation_error'

    fig = plt.figure(dpi=300)
    fig.set_size_inches(8, 1.5)
    fig.subplots_adjust(hspace=0.4)
    n, settings_dict = parse_dir(repo)
    freq = int(n/20)
    for j, eps in enumerate(settings_dict['eps']):
        ax = fig.add_subplot(1, len(settings_dict['eps']), j + 1)
        for i, exp in enumerate(settings_dict['exp']):            
            datas = [pd.read_csv(data) for data in glob.glob(exp + '|*='+str(eps)+'|*=*'+'.csv')]
            plot_curve(
                ax, 
                datas,
                color = COLORS[i],
                label = exp.split('/')[-1], 
                feature = label, 
                freq = freq,
                markersize = 1
            )
        ax.set_title(f"Privacy Epsilon={eps}")
        ax.set_xlabel("Timestep")
        if j == 0:
            ax.set_ylabel("Cumulative Regret")
        if j == len(settings_dict['eps']) - 1:
            ax.legend(loc="best", frameon=False)
    sns.despine()
    # plt.show()
    tikzplotlib.save(repo + "/SingleRegret.tex")
    fig.savefig(repo + '/SingleRegret.pdf', dpi=300, bbox_inches='tight')

def glm_exp_plot(repo):
    COLORS = sns.color_palette("tab10")

    label = 'cum_regrets' #'cum_regrets' #'estimation_error'

    fig = plt.figure(dpi=300)
    fig.set_size_inches(8, 3)
    fig.subplots_adjust(hspace=0.4)
    n, settings_dict = parse_dir(repo)
    freq = int(n/20)
    rewards = settings_dict['reward']
    epss = settings_dict['eps']
    print(epss)
    for k, reward in enumerate(rewards):
        for j, eps in enumerate(epss):
            ax = fig.add_subplot(2, len(epss), k*len(epss) + (j + 1))
            for i, exp in enumerate(settings_dict['exp']):            
                datas = [pd.read_csv(data) for data in glob.glob(exp + '|*='+str(eps)+ '|*='+str(reward) + '|*=*'+'.csv')]
                plot_curve(
                    ax, 
                    datas,
                    color = COLORS[i],
                    label = exp.split('/')[-1], 
                    feature = label,
                    freq = freq,
                    markersize = 1
                )
            ax.set_title(f"Reward={reward}/Epsilon={eps}")
            ax.set_xlabel("Timestep")
            if j == 0:
                ax.set_ylabel("Cumulative Regret")
            if j == len(settings_dict['eps']) - 1:
                ax.legend(loc="best", frameon=False)
    plt.subplots_adjust(wspace = 0.3, hspace = 0.8)
    sns.despine()
    # plt.show()
    tikzplotlib.save(repo + "/GeneralizedLinearRegret.tex")
    fig.savefig(repo + '/GeneralizedLinearRegret.pdf', dpi=300, bbox_inches='tight')
    
def crpm_exp_plot(repo):
    COLORS = sns.color_palette("tab10")

    label = 'cum_regrets' #'cum_regrets' #'estimation_error'

    fig = plt.figure(dpi=300)
    fig.set_size_inches(8, 1.5)
    fig.subplots_adjust(hspace=0.4)
    n, settings_dict = parse_dir(repo)
    freq = int(n/20)
    for j, eps in enumerate(settings_dict['eps']):
        ax = fig.add_subplot(1, len(settings_dict['eps']), j + 1)
        for i, exp in enumerate(settings_dict['exp']):            
            datas = [pd.read_csv(data) for data in glob.glob(exp + '|*='+str(eps)+'|*=*'+'.csv')]
            plot_curve(
                ax, 
                datas,
                color = COLORS[i],
                label = exp.split('/')[-1], 
                feature = label, 
                freq = freq,
                markersize = 1
            )
        ax.set_title(f"Privacy Epsilon={eps}")
        ax.set_xlabel("Timestep")
        if j == 0:
            ax.set_ylabel("Cumulative Regret")
        if j == len(settings_dict['eps']) - 1:
            ax.legend(loc="best", frameon=False)
    sns.despine()
    # plt.show()
    tikzplotlib.save(repo + "/CrpmRegret.tex")
    fig.savefig(repo + '/CrpmRegret.pdf', dpi=300, bbox_inches='tight')

def multiparam_exp_plot(repo, title):
    COLORS = sns.color_palette("tab10")

    label = 'cum_regrets' #'cum_regrets' #'estimation_error'

    fig = plt.figure(dpi=300)
    fig.set_size_inches(8, 3)
    fig.subplots_adjust(hspace=0.4)
    n, settings_dict = parse_dir(repo)
    freq = int(n/20)
    n_actions = settings_dict['n_action']
    epss = settings_dict['eps']
    for k, n_action in enumerate(n_actions):
        for j, eps in enumerate(epss):
            ax = fig.add_subplot(2, len(epss), k*(len(n_actions) + 1) + (j + 1))
            for i, exp in enumerate(settings_dict['exp']):  
                datas = [pd.read_csv(data) for data in glob.glob(exp + '|*='+str(eps)+ '|*='+str(n_action) + '|*=*'+'.csv')]
                plot_curve(
                    ax, 
                    datas,
                    color = COLORS[i],
                    label = 'LDP-SGD', 
                    feature = 'sgdr', 
                    freq = freq,
                    markersize = 1
                )
                plot_curve(
                    ax, 
                    datas,
                    color = COLORS[i+1],
                    label = 'LDP-OLS', 
                    feature = 'olsr', 
                    freq = freq,
                    markersize = 1
                )
                plot_curve(
                    ax, 
                    datas,
                    color = COLORS[i+2],
                    label = 'LDP-UCB', 
                    feature = 'ucbr', 
                    freq = freq,
                    markersize = 1
                )
                plot_curve(
                    ax, 
                    datas,
                    color = COLORS[i+3],
                    label = 'LDP-GLOC', 
                    feature = 'glocr', 
                    freq = freq,
                    markersize = 1
                )
            ax.set_title(f"Epsilon={eps}")
            ax.set_xlabel("Timestep")
            if j == 0:
                ax.set_ylabel("Cumulative Regret")
            if j == len(settings_dict['eps']) - 1:
                ax.legend(loc="best", frameon=False)
    plt.subplots_adjust(wspace = 0.3, hspace = 0.8)
    sns.despine()
    # plt.show()
    tikzplotlib.save(repo + "/" + title + ".tex")
    fig.savefig(repo + "/" + title + ".pdf", dpi=300, bbox_inches='tight')

def ees_exp_plot(repo, title):
    COLORS = sns.color_palette("tab10")

    label = 'cum_regrets' #'cum_regrets' #'estimation_error'

    fig = plt.figure(dpi=300)
    fig.set_size_inches(8, 3)
    fig.subplots_adjust(hspace=0.4)
    n, settings_dict = parse_dir(repo)
    freq = int(n/20)
    n_actions = settings_dict['n_action']
    epss = settings_dict['eps']
    for k, n_action in enumerate(n_actions):
        for j, eps in enumerate(epss):
            ax = fig.add_subplot(2, len(epss), k*(len(n_actions) + 1) + (j + 1))
            for i, exp in enumerate(settings_dict['exp']):  
                datas = [pd.read_csv(data) for data in glob.glob(exp + '|*='+str(eps)+ '|*='+str(n_action) + '|*=*'+'.csv')]
                plot_curve(
                    ax, 
                    datas,
                    color = COLORS[i],
                    label = 'LDP-SGD', 
                    feature = 'sgdr', 
                    freq = freq,
                    markersize = 1
                )
                plot_curve(
                    ax, 
                    datas,
                    color = COLORS[i+1],
                    label = 'LDP-OLS', 
                    feature = 'olsr', 
                    freq = freq,
                    markersize = 1
                )
                plot_curve(
                    ax, 
                    datas,
                    color = COLORS[i+2],
                    label = 'LDP-UCB', 
                    feature = 'ucbr', 
                    freq = freq,
                    markersize = 1
                )
            ax.set_title(f"Epsilon={eps}")
            ax.set_xlabel("Timestep")
            if j == 0:
                ax.set_ylabel("Cumulative Regret")
            if j == len(settings_dict['eps']) - 1:
                ax.legend(loc="best", frameon=False)
    plt.subplots_adjust(wspace = 0.3, hspace = 0.8)
    sns.despine()
    # plt.show()
    tikzplotlib.save(repo + "/" + title + ".tex")
    fig.savefig(repo + "/" + title + ".pdf", dpi=300, bbox_inches='tight')