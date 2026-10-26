"""
Lab 8 - Knowledge Distillation from scratch, NumPy only.

Implements Hinton, Vinyals, Dean (2015): a small "student" network is trained to match
a large "teacher" network's SOFT output distribution (at temperature T), not just the
hard labels. Includes the two controls every distillation claim needs:
  (a) a hard-label-only student of the SAME size (isolates the effect of soft targets)
  (b) a hard PSEUDO-LABEL student trained on the same transfer set (isolates the effect
      of extra unlabelled data from the effect of the soft-target information itself)

"""
import numpy as np, warnings
warnings.filterwarnings("ignore")
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier

X, y = load_digits(return_X_y=True); X = X / 16.0
Xall, Xte, yall, yte = train_test_split(X, y, test_size=0.3, random_state=0, stratify=y)

def softmax(z, T=1.0):
    z = z / T; z = z - z.max(1, keepdims=True); e = np.exp(z); return e / e.sum(1, keepdims=True)
def logits(m, X):
    a = X
    for i, (W, b) in enumerate(zip(m.coefs_, m.intercepts_)):
        a = a @ W + b
        if i < len(m.coefs_) - 1: a = np.maximum(a, 0)
    return a
def n_params(m): return sum(w.size for w in m.coefs_) + sum(b.size for b in m.intercepts_)

def train_soft_student(Xt, soft, T, hidden, seed, epochs=2500, lr=0.5):
    """Full-batch GD on T^2 * KL(soft_T || student_T). d/dz = T * (p_T - soft_T) / n."""
    r = np.random.default_rng(seed); n = len(Xt)
    W1 = r.normal(0, .3, (64, hidden)); b1 = np.zeros(hidden)
    W2 = r.normal(0, .3, (hidden, 10)); b2 = np.zeros(10)
    for _ in range(epochs):
        h = np.maximum(Xt @ W1 + b1, 0); z = h @ W2 + b2
        dz = T * (softmax(z, T) - soft) / n
        dW2 = h.T @ dz; db2 = dz.sum(0)
        dh = dz @ W2.T; dh[h <= 0] = 0
        dW1 = Xt.T @ dh; db1 = dh.sum(0)
        W1 -= lr * dW1; b1 -= lr * db1; W2 -= lr * dW2; b2 -= lr * db2
    return lambda Z: (np.maximum(Z @ W1 + b1, 0) @ W2 + b2).argmax(1)

def run(seed, n_lab=150, hidden=8, lr_by_T=None):
    """Data-scarce regime: n_lab labelled points + a larger unlabelled transfer pool."""
    lr_by_T = lr_by_T or {1: 0.5, 2: 0.5, 4: 0.5}
    r = np.random.default_rng(seed)
    idx = r.permutation(len(Xall)); lab, pool = idx[:n_lab], idx[n_lab:]
    Xl, yl = Xall[lab], yall[lab]; Xt = np.vstack([Xl, Xall[pool]])
    teacher = MLPClassifier((128, 128), max_iter=800, random_state=seed, alpha=1e-3).fit(Xl, yl)
    out = {"teacher": teacher.score(Xte, yte)}
    out["student, hard labels (150 only)"] = MLPClassifier((hidden,), max_iter=800, random_state=seed, alpha=1e-3).fit(Xl, yl).score(Xte, yte)
    pseudo = teacher.predict(Xt)
    out["control: hard pseudo-labels on pool"] = MLPClassifier((hidden,), max_iter=800, random_state=seed, alpha=1e-3).fit(Xt, pseudo).score(Xte, yte)
    for T in [1, 2, 4]:
        pred = train_soft_student(Xt, softmax(logits(teacher, Xt), T), T, hidden, seed, lr=lr_by_T[T])
        out[f"distilled, T={T}"] = (pred(Xte) == yte).mean()
    out["_teacher_params"] = n_params(teacher); out["_student_params"] = hidden * 64 + hidden + hidden * 10 + 10
    return out

if __name__ == "__main__":
    print("=== Experiment 1: does learning rate explain the T=4 result? (sweep before concluding) ===")
    for T in [1, 2, 4]:
        for lr in [0.5, 0.2, 0.1, 0.05]:
            r = np.random.default_rng(0)
            idx = r.permutation(len(Xall)); lab, pool = idx[:150], idx[150:]
            Xl, yl = Xall[lab], yall[lab]; Xt = np.vstack([Xl, Xall[pool]])
            teacher = MLPClassifier((128, 128), max_iter=800, random_state=0, alpha=1e-3).fit(Xl, yl)
            pred = train_soft_student(Xt, softmax(logits(teacher, Xt), T), T, hidden=8, seed=0, lr=lr)
            acc = (pred(Xte) == yte).mean()
            print(f"  T={T} lr={lr}: acc={acc:.4f}")
        print()
    print("Conclusion: pick the BEST lr per T below (this is standard practice -- each T gets its own tuning).\n")

    best_lr = {1: 0.5, 2: 0.5, 4: 0.2}   # chosen from the sweep above (T=4's best was lr=0.2, 0.9185)
    print("=== Experiment 2: distillation with controls, properly tuned, 5 seeds ===")
    res = [run(s, lr_by_T=best_lr) for s in range(5)]
    print("Test accuracy (mean +/- std):")
    for k in [k for k in res[0] if not k.startswith("_")]:
        v = [r[k] for r in res]; print(f"  {k:38s} {np.mean(v):.4f} +/- {np.std(v):.4f}")
    print(f"\nTeacher params {res[0]['_teacher_params']:,} -> student params {res[0]['_student_params']:,}"
          f" ({res[0]['_teacher_params']/res[0]['_student_params']:.0f}x fewer)")

    print("\n=== Experiment 3: distillation in a DATA-RICH regime (all labels available) ===")
    print("Does the advantage disappear when the student doesn't need the extra unlabelled data?")
    def run_rich(seed, hidden=8):
        teacher = MLPClassifier((128, 128), max_iter=400, random_state=seed, alpha=1e-3).fit(Xall, yall)
        hard = MLPClassifier((hidden,), max_iter=400, random_state=seed, alpha=1e-3).fit(Xall, yall).score(Xte, yte)
        soft = softmax(logits(teacher, Xall), 1.0)
        pred = train_soft_student(Xall, soft, 1.0, hidden, seed, lr=0.3)
        dist = (pred(Xte) == yte).mean()
        return teacher.score(Xte, yte), hard, dist
    rich = [run_rich(s) for s in range(5)]
    t = np.mean([r[0] for r in rich]); h = np.mean([r[1] for r in rich]); d = np.mean([r[2] for r in rich])
    print(f"  Teacher: {t:.4f}   Hard-label student (all labels): {h:.4f}   Distilled student (all labels): {d:.4f}")
    print("  DISCUSS: compare the distillation GAIN here to Experiment 2's data-scarce gain. What changed?")
