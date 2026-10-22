# Kirman's ant model: flipping crowds, simulated and compared with theory

Ants given two identical food sources don't split 50/50. Most pile onto one source,
then the colony suddenly flips to the other. Alan Kirman modelled this as a herding
process, and it is also used as a toy model of financial markets.

This repo simulates the model agent by agent and compares the result with its
continuous (diffusion / Fokker-Planck) description.

## The model

`N` agents each hold opinion A or B. Every step, one agent is picked and may change:

- **Random switch** at rate `eps`: flips on its own (noise).
- **Copy** at rate `h`: adopts the opinion of a randomly chosen other agent (herding).

With `k` agents in camp A and `x = k/N`:

```
P(k -> k+1) = (1 - x) * (eps + h * k / (N-1))
P(k -> k-1) =      x  * (eps + h * (N-k) / (N-1))
```

## The diffusion limit

For large N, one step as one unit of time, the process becomes a stochastic
differential equation with

```
drift     a(x)  = eps * (1 - 2x) / N
variance  s2(x) = (eps + 2 h x (1 - x)) / N^2
```

whose stationary density is

```
p(x)  ~  (eps + 2 h x (1 - x)) ^ (eps*N/h - 1)
```

The key control parameter is **eps*N/h**:

| eps*N/h | shape of p(x) | behaviour |
|---|---|---|
| < 1 | U-shaped (bimodal) | colony sits near consensus and flips between A and B |
| = 1 | flat | critical point |
| > 1 | peaked at 1/2 | noise wins, no herding |

Note the threshold depends on `eps*N/h`, not `eps/h`: bigger groups need less
noise to stay noise-dominated. (A Beta distribution is a common approximation of
this density when eps is small, but the form above is what the derivation gives.)

## Run it

```bash
pip install -r requirements.txt
python kirman_ants.py                       # writes 4 figures to ./figures
python kirman_ants.py --N 200 --eps 0.001 --h 1 --steps 3000000
```

Figures produced: `bimodal_flipping.png`, `critical.png`, `unimodal_noisy.png`,
and `custom_run.png` for your own parameters. Each shows `x(t)` on the left and the
simulated histogram against the diffusion prediction (red) on the right.

## Reading the plots

- In the bimodal regime the time series jumps between near 0 and near 1 and
  rarely rests at 0.5.
- The theory curve diverges at x = 0 and 1, while the simulation has finite bars
  there because `k` is an integer. Away from the edges the two agree.

## Files

- `kirman_ants.py`: simulation, theory, plotting
- `requirements.txt`: numpy, matplotlib
- `figures/`: example output
