"""
Reproduce the diversity experiment (Figures comparing epoch greedy, greedy, greedy first).
Two scenarios: without strong diversity (non-diverse) and with strong diversity (diverse).

Usage:
    python reproduce_diversity.py          # show plots
    python reproduce_diversity.py --save   # save to experiments_diversity/{nd,d}{1,5,10}.png
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

# ---------------- Parameters ----------------
d, T = 3, 10_000
N_simu = 20
eps_list = [0.1, 0.5, 1.0]
noise_std = 0.1
alpha_gd = lambda k: 1.0 / k
np.random.seed(42)

M = 50
W = T // M
cov_const = 1e-4

theta_star = np.array([5.0 + 4 / np.sqrt(T), 3.0, 5.0])


def gen_TS(horizon):
    t, i, sched = 1, 1, []
    while t <= horizon:
        sched.append(t)
        t += int(np.ceil(np.sqrt(i)))
        i += 1
    return sched


T_S_full = gen_TS(T)


def sample_unit_vec(dim=3):
    v = np.random.normal(size=dim)
    return v / np.linalg.norm(v)


def one_run_nondiv(eps):
    """Non-diverse context distribution."""
    theta_epoch = np.zeros(d)
    theta_greedy = np.zeros(d)
    theta_adapt = np.zeros(d)

    G_e = np.zeros(d)
    cov_e = np.zeros((d, d))
    epoch_id = 1
    switched = False
    TS_adapt = set()

    reg_E, reg_G, reg_A = [], [], []
    k_E = k_G = 1
    k_A_updates = 0

    for t in range(1, T + 1):
        ctx1 = np.random.uniform([1, 0, 0], [0, 1, 0])
        ctx2 = np.array([0, 0, 1])
        X = [ctx1, ctx2]
        tru = [x @ theta_star for x in X]

        # epoch greedy
        if t in T_S_full:
            aE = np.random.choice([0, 1])
            xE = X[aE]
            yE = xE @ theta_star + np.random.normal(0, noise_std)
            gradE = (theta_epoch @ xE - yE) * xE + np.random.normal(0, 1 / eps, d)
            theta_epoch -= alpha_gd(k_E) * gradE
            k_E += 1
        else:
            aE = int((X[1] @ theta_epoch) > (X[0] @ theta_epoch))
        reg_E.append(max(tru) - tru[aE])

        # greedy
        aG = int((X[1] @ theta_greedy) > (X[0] @ theta_greedy))
        xG = X[aG]
        yG = xG @ theta_star + np.random.normal(0, noise_std)
        gradG = (theta_greedy @ xG - yG) * xG + np.random.normal(0, 1 / eps, d)
        theta_greedy -= alpha_gd(k_G) * gradG
        k_G += 1
        reg_G.append(max(tru) - tru[aG])

        # greedy first
        if not switched:
            aA = int((X[1] @ theta_adapt) > (X[0] @ theta_adapt))
            xA = X[aA]
            yA = xA @ theta_star + np.random.normal(0, noise_std)
            G_e += (theta_adapt @ xA - yA) * xA + np.random.normal(0, 1 / eps, d)
            cov_e += np.outer(xA, xA)
            if (t % W == 0) or (t == T):
                theta_adapt -= (1 / epoch_id) * G_e
                epoch_id += 1
                if np.min(np.linalg.eigvalsh(cov_e)) < cov_const * W:
                    switched = True
                    theta_adapt = np.zeros(d)
                    k_A_updates = 1
                    TS_adapt = {t + s for s in gen_TS(T - t)}
                G_e[:] = 0
                cov_e[:] = 0
        else:
            if t in TS_adapt:
                aA = np.random.choice([0, 1])
                xA = X[aA]
                yA = xA @ theta_star + np.random.normal(0, noise_std)
                gradA = (theta_adapt @ xA - yA) * xA + np.random.normal(0, 1 / eps, d)
                theta_adapt -= alpha_gd(k_A_updates) * gradA
                k_A_updates += 1
            else:
                aA = int((X[1] @ theta_adapt) > (X[0] @ theta_adapt))
        reg_A.append(max(tru) - tru[aA])

    return np.cumsum(reg_E), np.cumsum(reg_G), np.cumsum(reg_A)


def one_run_div(eps):
    """Diverse context distribution (unit vectors)."""
    theta_epoch = np.zeros(d)
    theta_greedy = np.zeros(d)
    theta_adapt = np.zeros(d)

    G_e = np.zeros(d)
    cov_e = np.zeros((d, d))
    epoch_id = 1
    switched = False
    TS_adapt = set()

    reg_E, reg_G, reg_A = [], [], []
    k_E = k_G = 1
    k_A_updates = 0

    for t in range(1, T + 1):
        ctx1 = sample_unit_vec(d)
        ctx2 = sample_unit_vec(d)
        X = [ctx1, ctx2]
        true_r = [x @ theta_star for x in X]

        # epoch greedy
        if t in T_S_full:
            aE = np.random.choice([0, 1])
            xE = X[aE]
            yE = xE @ theta_star + np.random.normal(0, noise_std)
            gradE = (theta_epoch @ xE - yE) * xE + np.random.normal(0, 1 / eps, d)
            theta_epoch -= alpha_gd(k_E) * gradE
            k_E += 1
        else:
            aE = int((X[1] @ theta_epoch) > (X[0] @ theta_epoch))
        reg_E.append(max(true_r) - true_r[aE])

        # greedy
        aG = int((X[1] @ theta_greedy) > (X[0] @ theta_greedy))
        xG = X[aG]
        yG = xG @ theta_star + np.random.normal(0, noise_std)
        gradG = (theta_greedy @ xG - yG) * xG + np.random.normal(0, 1 / eps, d)
        theta_greedy -= alpha_gd(k_G) * gradG
        k_G += 1
        reg_G.append(max(true_r) - true_r[aG])

        # greedy first
        if not switched:
            aA = int((X[1] @ theta_adapt) > (X[0] @ theta_adapt))
            xA = X[aA]
            yA = xA @ theta_star + np.random.normal(0, noise_std)
            G_e += (theta_adapt @ xA - yA) * xA + np.random.normal(0, 1 / eps, d)
            cov_e += np.outer(xA, xA)
            if (t % W == 0) or (t == T):
                theta_adapt -= (1 / epoch_id) * G_e
                epoch_id += 1
                if np.min(np.linalg.eigvalsh(cov_e)) < cov_const * W:
                    switched = True
                    theta_adapt = np.zeros(d)
                    k_A_updates = 1
                    TS_adapt = {t + s for s in gen_TS(T - t)}
                G_e[:] = 0
                cov_e[:] = 0
        else:
            if t in TS_adapt:
                aA = np.random.choice([0, 1])
                xA = X[aA]
                yA = xA @ theta_star + np.random.normal(0, noise_std)
                gradA = (theta_adapt @ xA - yA) * xA + np.random.normal(0, 1 / eps, d)
                theta_adapt -= alpha_gd(k_A_updates) * gradA
                k_A_updates += 1
            else:
                aA = int((X[1] @ theta_adapt) > (X[0] @ theta_adapt))
        reg_A.append(max(true_r) - true_r[aA])

    return np.cumsum(reg_E), np.cumsum(reg_G), np.cumsum(reg_A)


def run_and_plot(run_fn, scenario_name, eps_list, save_prefix=None):
    """Run Monte Carlo simulations and plot for all epsilon values."""
    eps_file_map = {0.1: "1", 0.5: "5", 1.0: "10"}

    for eps in eps_list:
        print(f"  Running {N_simu} sims for eps={eps}...")
        all_E, all_G, all_A = [], [], []
        for _ in range(N_simu):
            rE, rG, rA = run_fn(eps)
            all_E.append(rE); all_G.append(rG); all_A.append(rA)

        mE, sE = np.mean(all_E, 0), np.std(all_E, 0)
        mG, sG = np.mean(all_G, 0), np.std(all_G, 0)
        mA, sA = np.mean(all_A, 0), np.std(all_A, 0)
        rounds = np.arange(1, T + 1)

        plt.figure(figsize=(9, 5))
        plt.plot(mE, label="epoch greedy", lw=1.8)
        plt.fill_between(rounds, mE - sE, mE + sE, alpha=0.2)
        plt.plot(mG, label="greedy", lw=1.8)
        plt.fill_between(rounds, mG - sG, mG + sG, alpha=0.2)
        plt.plot(mA, label="greedy first", lw=1.8)
        plt.fill_between(rounds, mA - sA, mA + sA, alpha=0.2)

        plt.title(rf"Cumulative regret ($\varepsilon$={eps}, {N_simu} runs, {scenario_name})")
        plt.xlabel("round")
        plt.ylabel("cumulative regret")
        plt.legend()
        plt.grid(True)
        plt.ylim(0, 6000)
        plt.xlim(0, T)
        plt.tight_layout()

        if save_prefix:
            outpath = f"experiments_diversity/{save_prefix}{eps_file_map[eps]}.png"
            plt.savefig(outpath, dpi=150)
            print(f"    Saved: {outpath}")
        else:
            plt.show()


if __name__ == "__main__":
    save_mode = "--save" in sys.argv
    save_pfx_nd = "nd" if save_mode else None
    save_pfx_d = "d" if save_mode else None

    print("=== Without strong diversity ===")
    run_and_plot(one_run_nondiv, "non-diverse", eps_list, save_pfx_nd)

    print("\n=== With strong diversity ===")
    run_and_plot(one_run_div, "diverse", eps_list, save_pfx_d)

    print("\nDone!")
