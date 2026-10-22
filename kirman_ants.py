"""
Kirman's ant model: agent-based simulation vs. diffusion (Fokker-Planck) limit.

N agents hold one of two opinions (A or B). Each step, one agent is picked:
  - with rate eps it switches on its own (random switching / noise)
  - with rate h it copies a randomly chosen other agent (herding)

k = number of agents in camp A, x = k / N.

Per-step transition probabilities:
  P(k -> k+1) = (1 - k/N) * (eps + h * k / (N-1))
  P(k -> k-1) = (k/N)     * (eps + h * (N-k) / (N-1))

Diffusion limit (large N), one step = one unit of time:
  drift      a(x)  = eps * (1 - 2x) / N
  variance   s2(x) = (eps + 2h x (1-x)) / N^2
Stationary density of the resulting Fokker-Planck equation:
  p(x) ~ (eps + 2h x (1-x)) ** (eps*N/h - 1)
  -> bimodal (U-shaped, flipping ants) if eps*N/h < 1
  -> unimodal (peak at 1/2)            if eps*N/h > 1

Usage:
  python kirman_ants.py                 # make all figures into ./figures
  python kirman_ants.py --N 200 --eps 0.001 --h 1 --steps 3000000
"""
import argparse
import os

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def simulate(N=100, eps=0.002, h=1.0, steps=2_000_000, seed=0, thin=100):
    """Run the agent-level Markov chain. Returns x(t) sampled every `thin` steps
    and the full list of visited k values (for the histogram)."""
    rng = np.random.default_rng(seed)
    u = rng.random(steps)
    k = N // 2
    ks = np.empty(steps, dtype=np.int32)
    for t in range(steps):
        p_up = (1 - k / N) * (eps + h * k / (N - 1))
        p_down = (k / N) * (eps + h * (N - k) / (N - 1))
        r = u[t]
        if r < p_up:
            k += 1
        elif r < p_up + p_down:
            k -= 1
        ks[t] = k
    return ks[::thin] / N, ks / N


_trapz = getattr(np, "trapezoid", None) or np.trapz


def stationary_density(x, N, eps, h):
    """Stationary density of the diffusion approximation (normalised numerically)."""
    p = (eps + 2 * h * x * (1 - x)) ** (eps * N / h - 1)
    return p / _trapz(p, x)


def make_regime_figure(N, eps, h, steps, seed, label, path):
    x_t, x_all = simulate(N, eps, h, steps, seed)
    ratio = eps * N / h
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.6), gridspec_kw={"width_ratios": [2, 1]})

    show = min(len(x_t), 20000)
    ax[0].plot(np.arange(show), x_t[:show], lw=0.6)
    ax[0].axhline(0.5, color="gray", ls=":", lw=0.8)
    ax[0].set_ylim(0, 1)
    ax[0].set_xlabel(f"time (x{100} steps)")
    ax[0].set_ylabel("fraction in camp A, x")
    ax[0].set_title(f"{label}: x(t)   (eps*N/h = {ratio:.2f})")

    ax[1].hist(x_all, bins=np.linspace(0, 1, N // 2 + 1), density=True,
               alpha=0.6, label="simulation")
    xs = np.linspace(0.0, 1.0, 1000)
    ax[1].plot(xs, stationary_density(xs, N, eps, h), "r", lw=2, label="diffusion theory")
    ax[1].set_xlabel("x")
    ax[1].set_ylabel("density")
    ax[1].set_title("distribution of x")
    ax[1].legend()
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--N", type=int, default=100)
    ap.add_argument("--eps", type=float, default=0.002)
    ap.add_argument("--h", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=2_000_000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="figures")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    # Single run with user parameters
    make_regime_figure(args.N, args.eps, args.h, args.steps, args.seed,
                       "custom run", os.path.join(args.out, "custom_run.png"))

    # Three regimes side by side: flipping, critical, noise-dominated
    for name, ratio in [("bimodal_flipping", 0.2), ("critical", 1.0), ("unimodal_noisy", 4.0)]:
        eps = ratio * args.h / args.N
        make_regime_figure(args.N, eps, args.h, args.steps, args.seed,
                           name.replace("_", " "), os.path.join(args.out, f"{name}.png"))
    print(f"Figures written to {args.out}/")


if __name__ == "__main__":
    main()
