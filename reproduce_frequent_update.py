"""
Reproduce the frequent-update experiment (Figure comparing Algorithm 1 vs Algorithm 5).
Generates 4 plots for epsilon = 0.1, 0.5, 1.0, +inf, each showing mean +/- std
of cumulative regret over N_simu simulation runs.

Usage:
    python reproduce_frequent_update.py          # show plots
    python reproduce_frequent_update.py --save   # save to new_pics/pic{1..4}.png
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

# Parameters
d = 3
T = 10000
N_simu = 20
epsilon_list = [0.1, 0.5, 1.0, 1000.0]
noise_std = 0.1
alpha_schedule = lambda k: 1.0 / k
np.random.seed(42)


def generate_TS(T):
    """Generate the sparse update time index set T_S."""
    T_S = []
    t = 1
    i = 1
    while t <= T:
        T_S.append(t)
        t = t + int(np.ceil(np.sqrt(i)))
        i += 1
    return set(T_S)


T_S_set = generate_TS(T)


def run_simulation(T, epsilon):
    """
    Run one simulation trial.
    Algorithm 1: updates theta only at times in T_S (sparse update).
    Algorithm 5: updates theta at every time step (frequent update).
    Returns cumulative regret arrays for both algorithms.
    """
    theta_star = np.array([5.0 + 4 / np.sqrt(T), 3.0, 5.0])
    theta_1 = np.zeros(d)
    theta_2 = np.zeros(d)
    regret_1 = []
    regret_2 = []
    k1 = 1
    k2 = 1

    for t in range(1, T + 1):
        xt1 = np.random.uniform([1, 0, 0], [0, 1, 0])
        xt2 = np.array([0, 0, 1])
        X = [xt1, xt2]

        # Algorithm 1: explore at T_S, exploit otherwise
        if t in T_S_set:
            action1 = np.random.choice([0, 1])
            x_chosen1 = X[action1]
            y1 = np.dot(x_chosen1, theta_star) + np.random.normal(0, noise_std)
            pred_y1 = np.dot(theta_1, x_chosen1)
            grad1 = (pred_y1 - y1) * x_chosen1 + np.random.normal(0, 1 / epsilon, size=d)
            theta_1 -= alpha_schedule(k1) * grad1
            k1 += 1
        else:
            pred_rewards1 = [np.dot(x, theta_1) for x in X]
            action1 = np.argmax(pred_rewards1)

        true_rewards = [np.dot(x, theta_star) for x in X]
        regret_1.append(max(true_rewards) - true_rewards[action1])

        # Algorithm 5: explore at T_S, but always update
        if t in T_S_set:
            action2 = np.random.choice([0, 1])
        else:
            pred_rewards2 = [np.dot(x, theta_2) for x in X]
            action2 = np.argmax(pred_rewards2)

        x_chosen2 = X[action2]
        y2 = np.dot(x_chosen2, theta_star) + np.random.normal(0, noise_std)
        pred_y2 = np.dot(theta_2, x_chosen2)
        grad2 = (pred_y2 - y2) * x_chosen2 + np.random.normal(0, 1 / epsilon, size=d)
        theta_2 -= alpha_schedule(k2) * grad2
        k2 += 1

        regret_2.append(max(true_rewards) - true_rewards[action2])

    return np.cumsum(regret_1), np.cumsum(regret_2)


if __name__ == "__main__":
    save_mode = "--save" in sys.argv

    eps_labels = {0.1: "0.1", 0.5: "0.5", 1.0: "1", 1000.0: r"+\infty"}
    eps_filenames = {0.1: "pic1", 0.5: "pic2", 1.0: "pic3", 1000.0: "pic4"}

    for eps in epsilon_list:
        print(f"Running {N_simu} simulations for epsilon = {eps}...")
        regrets1 = []
        regrets2 = []
        for _ in range(N_simu):
            r1, r2 = run_simulation(T, eps)
            regrets1.append(r1)
            regrets2.append(r2)

        regrets1 = np.array(regrets1)
        regrets2 = np.array(regrets2)
        mean_r1 = regrets1.mean(axis=0)
        std_r1 = regrets1.std(axis=0)
        mean_r2 = regrets2.mean(axis=0)
        std_r2 = regrets2.std(axis=0)

        plt.figure(figsize=(8, 5))
        t_axis = np.arange(T)
        plt.plot(t_axis, mean_r1, label="Algorithm 1", linestyle="-")
        plt.fill_between(t_axis, mean_r1 - std_r1, mean_r1 + std_r1, alpha=0.2)
        plt.plot(t_axis, mean_r2, label="Algorithm 5", linestyle="-")
        plt.fill_between(t_axis, mean_r2 - std_r2, mean_r2 + std_r2, alpha=0.2)
        plt.title(
            rf"Avg. Cumulative Regret $\pm$ Std ($\varepsilon = {eps_labels[eps]}$, {N_simu} runs)"
        )
        plt.xlabel("Time")
        plt.ylabel("Cumulative Regret")
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 5000)
        plt.xlim(0, T)
        plt.tight_layout()

        if save_mode:
            outpath = f"new_pics/{eps_filenames[eps]}.png"
            plt.savefig(outpath, dpi=150)
            print(f"  Saved: {outpath}")
        else:
            plt.show()

    print("Done!")
