"""
Lab 10 - A toy hardware-aware Neural Architecture Search, from scratch.

Full NAS (reinforcement learning controllers, weight-sharing supernets, evolutionary
population search over millions of candidates) needs a GPU cluster and days of compute --
completely infeasible for a classroom lab. This lab instead implements the CORE IDEA at a
scale that runs on a laptop in minutes: a small, well-defined search space; a MULTI-OBJECTIVE
reward combining accuracy AND a hardware cost proxy (exactly as MnasNet, Tan et al. 2019,
does); and three search strategies compared head-to-head (random search, a hill-climbing
local search, and an evolutionary search) so students see that the SEARCH STRATEGY matters,
not just the search SPACE.

Search space: a 3-block MLP-style network (standing in for a 3-stage CNN) where each block's
width is chosen from a small discrete set. This keeps each candidate trainable in well under
a second, so an exhaustive sweep is possible and can be used as GROUND TRUTH to check whether
the search strategies actually find good architectures.
"""
import numpy as np, itertools, time, warnings
warnings.filterwarnings("ignore")
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

X, y = load_digits(return_X_y=True); X = X / 16.0
Xtr, Xte, ytr, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

WIDTH_CHOICES = [4, 8, 16, 32, 64]   # the search space: width of each of 3 blocks
LATENCY_TARGET_MS = 0.05             # a toy hardware budget

def architecture_cost(widths):
    """Analytic proxy: parameter count (as a MAC/latency stand-in), matching the Lab 1/9
    style of estimating cost WITHOUT running the model -- this is what makes search fast."""
    layers = [64] + list(widths) + [10]
    params = sum(layers[i] * layers[i+1] + layers[i+1] for i in range(len(layers) - 1))
    return params

def train_and_eval(widths, seed=0):
    """The EXPENSIVE part of NAS: actually train the candidate and measure accuracy."""
    m = MLPClassifier(hidden_layer_sizes=tuple(widths), max_iter=150, random_state=seed, alpha=1e-3)
    m.fit(Xtr, ytr)
    return m.score(Xte, yte)

def reward(acc, params, param_budget, w=-0.07):
    """MnasNet-style weighted reward: accuracy * (params/budget)^w. w<0 penalizes exceeding
    the budget; the exponent (not a hard cutoff) lets a big accuracy gain still win sometimes --
    exactly the soft multi-objective trade-off real hardware-aware NAS uses."""
    return acc * ((params / param_budget) ** w)

PARAM_BUDGET = 3000   # a toy "edge device" parameter budget

# ---------------------------------------------------------------- ground truth (exhaustive)
def exhaustive_search(seed=0):
    results = []
    for widths in itertools.product(WIDTH_CHOICES, repeat=3):
        p = architecture_cost(widths)
        acc = train_and_eval(widths, seed=seed)
        results.append((widths, acc, p, reward(acc, p, PARAM_BUDGET)))
    return results

# ---------------------------------------------------------------- strategy 1: random search
def random_search(n_trials, seed=0):
    r = np.random.default_rng(seed)
    best = None
    trace = []
    for _ in range(n_trials):
        widths = tuple(r.choice(WIDTH_CHOICES, size=3))
        p = architecture_cost(widths)
        acc = train_and_eval(widths, seed=seed)
        rw = reward(acc, p, PARAM_BUDGET)
        if best is None or rw > best[3]: best = (widths, acc, p, rw)
        trace.append(best[3])
    return best, trace

# ---------------------------------------------------------------- strategy 2: hill climbing
def hill_climbing(n_trials, seed=0):
    r = np.random.default_rng(seed)
    cur = tuple(r.choice(WIDTH_CHOICES, size=3))
    cur_acc = train_and_eval(cur, seed=seed)
    cur_reward = reward(cur_acc, architecture_cost(cur), PARAM_BUDGET)
    best = (cur, cur_acc, architecture_cost(cur), cur_reward)
    trace = [best[3]]
    used = 1
    while used < n_trials:
        i = r.integers(0, 3)
        new_width = r.choice(WIDTH_CHOICES)
        cand = list(cur); cand[i] = new_width; cand = tuple(cand)
        acc = train_and_eval(cand, seed=seed); used += 1
        p = architecture_cost(cand)
        rw = reward(acc, p, PARAM_BUDGET)
        if rw > cur_reward:
            cur, cur_acc, cur_reward = cand, acc, rw
            if rw > best[3]: best = (cand, acc, p, rw)
        trace.append(best[3])
    return best, trace

# ---------------------------------------------------------------- strategy 3: evolutionary
def evolutionary_search(n_trials, pop_size=6, seed=0):
    r = np.random.default_rng(seed)
    pop = [tuple(r.choice(WIDTH_CHOICES, size=3)) for _ in range(pop_size)]
    scored = []
    used = 0
    for ind in pop:
        acc = train_and_eval(ind, seed=seed); used += 1
        scored.append((ind, acc, architecture_cost(ind), reward(acc, architecture_cost(ind), PARAM_BUDGET)))
    best = max(scored, key=lambda t: t[3])
    trace = [best[3]] * pop_size
    while used < n_trials:
        scored.sort(key=lambda t: -t[3])
        parent = scored[r.integers(0, max(2, pop_size // 2))][0]   # tournament-ish: pick from top half
        child = list(parent)
        i = r.integers(0, 3); child[i] = r.choice(WIDTH_CHOICES)   # mutate one block
        child = tuple(child)
        acc = train_and_eval(child, seed=seed); used += 1
        p = architecture_cost(child); rw = reward(acc, p, PARAM_BUDGET)
        scored.append((child, acc, p, rw))
        scored = sorted(scored, key=lambda t: -t[3])[:pop_size]   # survival of the fittest
        if rw > best[3]: best = (child, acc, p, rw)
        trace.append(best[3])
    return best, trace

if __name__ == "__main__":
    print(f"Search space: {len(WIDTH_CHOICES)}^3 = {len(WIDTH_CHOICES)**3} candidate architectures")
    print(f"Parameter budget: {PARAM_BUDGET} (toy edge-device proxy)\n")

    print("=== Ground truth: exhaustive search over the whole space ===")
    t0 = time.time()
    all_results = exhaustive_search()
    print(f"  Evaluated all {len(all_results)} architectures in {time.time()-t0:.1f}s")
    best_overall = max(all_results, key=lambda t: t[3])
    print(f"  BEST by reward: widths={best_overall[0]} acc={best_overall[1]:.4f} params={best_overall[2]} reward={best_overall[3]:.4f}")
    best_acc_only = max(all_results, key=lambda t: t[1])
    print(f"  BEST by accuracy alone (ignoring budget): widths={best_acc_only[0]} acc={best_acc_only[1]:.4f} params={best_acc_only[2]}")
    print("  DISCUSS: are these the same architecture? If not, what does that tell you about")
    print("           optimizing for accuracy alone vs optimizing for the deployment reward?\n")

    print("=== Search strategies, same budget of 20 trained candidates, 3 seeds each ===")
    n_trials = 20
    for name, fn in [("Random search", random_search), ("Hill climbing", hill_climbing), ("Evolutionary (pop=6)", evolutionary_search)]:
        finals = []
        for seed in range(3):
            best, trace = fn(n_trials, seed=seed)
            finals.append(best[3])
        gap = best_overall[3] - np.mean(finals)
        print(f"  {name:22s}: final reward = {np.mean(finals):.4f} +/- {np.std(finals):.4f}  "
              f"(gap to true optimum: {gap:.4f}, using {n_trials} of {len(all_results)} evaluations = "
              f"{100*n_trials/len(all_results):.0f}% of the space)")

    print("\nDISCUSS: NAS in research papers runs thousands of candidates on a cluster for days.")
    print("         This lab runs 20-125 candidates on a laptop in seconds. What does NOT scale")
    print("         down (besides raw compute) when you shrink a NAS experiment this much?")
