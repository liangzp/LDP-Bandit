"""
Reproduce Figure 3: Elliptical estimation error ||theta_tilde - theta*||_{M_t}
under adversarial vs stochastic context sequences.

Implements the Sequential LDP OLS Estimator (Eq. 122 in discussion_section.tex):
    M_t = M_{t-1} + phi_t phi_t^T + W_t       (W_t ~ Wigner)
    u_t = u_{t-1} + phi_t * r_t + w_t          (w_t ~ N(0, sigma_w^2 I))
    theta_tilde_t = (M_t + c_t I)^{-1} u_t

Privacy calibration (Proposition 5):
    sigma_w^2 = 8 log(1.25/delta) / epsilon^2
    sigma_W^2 = 4 C_B^2 sigma_w^2
    c_t = 2 sigma_W (4 sqrt(d) + 4 log(2T))

Usage:
    python reproduce_adv_context.py          # show plots
    python reproduce_adv_context.py --save   # save to adversary_context.png / stochastic_context.png
"""
import sys
import numpy as np
import matplotlib.pyplot as plt

# ============== Parameters ==============
d = 2
T = 100_000
N_simu = 30
epsilon = 5.0
delta = 1.0 / T
C_B = 1.0
noise_std = 0.1
theta_star = np.array([1.0, 1.0])

# Privacy noise calibration
sigma_w_sq = 8 * np.log(1.25 / delta) / epsilon**2
sigma_W_sq = 4 * C_B**2 * sigma_w_sq
sigma_W = np.sqrt(sigma_W_sq)
sigma_w = np.sqrt(sigma_w_sq)
# Regularization coefficient (c_t = c0 * sqrt(t) as in Eq. 168)
c0 = sigma_W

# Record at ~50 evenly spaced points
record_times = list(range(2000, T + 1, 2000))

np.random.seed(42)


def sample_wigner(d, sigma_sq):
    """Sample a d x d symmetric Wigner matrix.
    Off-diagonal (i<j): W_{ij} = W_{ji} ~ N(0, sigma^2).
    Diagonal: W_{ii} ~ N(0, 2*sigma^2).
    """
    W = np.zeros((d, d))
    for i in range(d):
        W[i, i] = np.random.normal(0, np.sqrt(2 * sigma_sq))
        for j in range(i + 1, d):
            val = np.random.normal(0, np.sqrt(sigma_sq))
            W[i, j] = val
            W[j, i] = val
    return W


def elliptical_error(theta_tilde, theta_star, M_reg):
    """Compute ||theta_tilde - theta*||_{M} = sqrt(diff^T M diff).
    Uses the regularized matrix to ensure PSD.
    """
    diff = theta_tilde - theta_star
    val = diff @ M_reg @ diff
    return np.sqrt(max(val, 0))


def run_ols_simulation(T, context_mode="stochastic"):
    """
    Run Sequential LDP OLS estimator.
    context_mode: "stochastic" or "adversarial"
    Returns: array of elliptical errors at record_times.
    """
    M_t = np.zeros((d, d))
    u_t = np.zeros(d)

    errors = []
    rec_idx = 0

    for t in range(1, T + 1):
        # Generate feature vector phi_t
        if context_mode == "adversarial":
            # Adversarial: always same direction — only accumulates rank-1 info
            phi_t = np.array([1.0, 0.0])
        else:
            # Stochastic: uniformly random on unit sphere (diverse)
            v = np.random.randn(d)
            phi_t = v / np.linalg.norm(v)

        # Observe reward
        r_t = phi_t @ theta_star + np.random.normal(0, noise_std)

        # Privacy noise
        W_t = sample_wigner(d, sigma_W_sq)
        w_t = np.random.normal(0, sigma_w, size=d)

        # OLS update
        M_t = M_t + np.outer(phi_t, phi_t) + W_t
        u_t = u_t + phi_t * r_t + w_t

        # Regularized estimate with c_t = c0 * sqrt(t)
        c_t = c0 * np.sqrt(t)
        M_reg = M_t + c_t * np.eye(d)
        theta_tilde = np.linalg.solve(M_reg, u_t)

        # Record at specified times
        if rec_idx < len(record_times) and t == record_times[rec_idx]:
            err = elliptical_error(theta_tilde, theta_star, M_reg)
            errors.append(err)
            rec_idx += 1

    return np.array(errors)


def run_experiment(context_mode, label):
    """Run N_simu simulations and return median and 80th percentile."""
    print(f"Running {N_simu} simulations ({label})...")
    all_errors = []
    for i in range(N_simu):
        if (i + 1) % 10 == 0:
            print(f"  sim {i+1}/{N_simu}")
        errs = run_ols_simulation(T, context_mode)
        all_errors.append(errs)
    all_errors = np.array(all_errors)  # (N_simu, len(record_times))
    median = np.median(all_errors, axis=0)
    quantile_80 = np.percentile(all_errors, 80, axis=0)
    return median, quantile_80


def plot_panel(ax, times, median, q80, ref_scale_t14, ref_scale_logt, title):
    """Plot one panel with median, 80th percentile, and reference curves.
    Reference curves use shared scaling constants (same scale for both panels).
    """
    t_arr = np.array(times, dtype=float)
    ref_t14 = ref_scale_t14 * t_arr**0.25
    ref_logt = ref_scale_logt * np.log(t_arr)

    ax.plot(times, median, label="median", color="C0", lw=1.5)
    ax.plot(times, q80, label="20-quantile", color="C1", lw=1.5)
    ax.plot(times, ref_t14, label=r"$T^{1/4}$", color="C2", ls="--", lw=1.5)
    ax.plot(times, ref_logt, label=r"$\log T$", color="C3", ls="--", lw=1.5)
    ax.set_title(title)
    ax.set_xlabel("t")
    ax.set_ylabel("elliptical distance")
    ax.legend()
    ax.grid(True)
    ax.set_xlim(0, T)
    ax.set_ylim(0, None)


if __name__ == "__main__":
    save_mode = "--save" in sys.argv

    # Run both experiments
    med_adv, q80_adv = run_experiment("adversarial", "adversarial context")
    med_sto, q80_sto = run_experiment("stochastic", "stochastic context")

    # Compute shared reference curve scaling from the adversarial median
    # (caption: "two figures are under the same scale")
    t_last = np.array(record_times, dtype=float)
    # Scale T^{1/4} to match adversarial median at the last time point
    ref_scale_t14 = med_adv[-1] / t_last[-1]**0.25
    # Scale log T similarly
    ref_scale_logt = ref_scale_t14 * t_last[-1]**0.25 / np.log(t_last[-1]) * 0.6

    # Set shared y-axis max (same scale, as stated in caption)
    ymax = max(np.max(q80_adv), np.max(q80_sto)) * 1.15

    # Plot adversarial
    fig1, ax1 = plt.subplots(figsize=(8, 6))
    plot_panel(ax1, record_times, med_adv, q80_adv,
               ref_scale_t14, ref_scale_logt,
               "Eillptical distance over 30 times of simulation")
    ax1.set_ylim(0, ymax)
    plt.tight_layout()
    if save_mode:
        fig1.savefig("adversary_context.png", dpi=150)
        print("Saved: adversary_context.png")
    else:
        plt.show()

    # Plot stochastic
    fig2, ax2 = plt.subplots(figsize=(8, 6))
    plot_panel(ax2, record_times, med_sto, q80_sto,
               ref_scale_t14, ref_scale_logt,
               "Eillptical distance over 30 times of simulation")
    ax2.set_ylim(0, ymax)
    plt.tight_layout()
    if save_mode:
        fig2.savefig("stochastic_context.png", dpi=150)
        print("Saved: stochastic_context.png")
    else:
        plt.show()

    print("Done!")
