# train.py — Builds a Random Forest from examples and saves it
# Run with: python train.py

import pickle
import os
import random
import math

# ── DATA ──────────────────────────────────────────────────────
# Each row: [my_hp, type_advantage, opp_hp, answer]
# type_advantage: 1 = yes, 0 = no

data = [
    [80, 1, 60, "Attack"],
    [70, 1, 50, "Attack"],
    [90, 1, 40, "Attack"],
    [60, 0, 20, "Attack"],   # opp almost dead → attack anyway
    [70, 0, 15, "Attack"],
    [55, 0, 80, "Switch"],   # no advantage + opp healthy → switch
    [80, 0, 90, "Switch"],
    [20, 0, 70, "Switch"],   # low HP → switch
    [15, 1, 60, "Switch"],
    [25, 0, 65, "Switch"],
]

X = [row[:3] for row in data]
y = [row[3]  for row in data]

FEATURE_NAMES = ["my_hp", "type_advantage", "opp_hp"]


# ── GINI — measures how mixed a group is (0 = pure, 0.5 = messy) ──
def gini(labels):
    total = len(labels)
    score = 1.0
    for label in set(labels):
        score -= (labels.count(label) / total) ** 2
    return score


# ── BOOTSTRAP SAMPLE — sample with replacement ──
def bootstrap_sample(X, y, seed=None):
    rng = random.Random(seed)
    n = len(X)
    indices = [rng.randint(0, n - 1) for _ in range(n)]
    return [X[i] for i in indices], [y[i] for i in indices]


# ── FIND BEST SPLIT — tries random subset of features ──
def best_split(X, y, n_features=None, rng=None):
    if rng is None:
        rng = random

    n_total_features = len(X[0])
    if n_features is None:
        n_features = max(1, int(math.sqrt(n_total_features)))

    # Randomly select a subset of features to consider (key to random forests)
    feature_indices = rng.sample(range(n_total_features), min(n_features, n_total_features))

    best = (None, None, 999.0)

    for feature in feature_indices:
        for threshold in set(row[feature] for row in X):

            left  = [y[i] for i in range(len(X)) if X[i][feature] <= threshold]
            right = [y[i] for i in range(len(X)) if X[i][feature] >  threshold]

            if not left or not right:
                continue

            score = (len(left) * gini(left) + len(right) * gini(right)) / len(y)

            if score < best[2]:
                best = (feature, threshold, score)

    return best[0], best[1]


# ── BUILD TREE — splits data recursively until groups are pure ──
def build(X, y, depth=0, max_depth=3, n_features=None, rng=None):
    if rng is None:
        rng = random

    # Stop if all answers are the same
    if len(set(y)) == 1:
        return {"answer": y[0], "samples": len(y), "depth": depth}

    # Stop if too deep or too few samples
    if depth >= max_depth or len(y) <= 1:
        majority = max(set(y), key=y.count)
        return {"answer": majority, "samples": len(y), "depth": depth}

    feature, threshold = best_split(X, y, n_features=n_features, rng=rng)

    if feature is None:
        majority = max(set(y), key=y.count)
        return {"answer": majority, "samples": len(y), "depth": depth}

    left_idx  = [i for i in range(len(X)) if X[i][feature] <= threshold]
    right_idx = [i for i in range(len(X)) if X[i][feature] >  threshold]

    left_X,  left_y  = [X[i] for i in left_idx],  [y[i] for i in left_idx]
    right_X, right_y = [X[i] for i in right_idx], [y[i] for i in right_idx]

    return {
        "feature"    : feature,
        "feature_name": FEATURE_NAMES[feature],
        "threshold"  : threshold,
        "left"       : build(left_X,  left_y,  depth + 1, max_depth, n_features, rng),
        "right"      : build(right_X, right_y, depth + 1, max_depth, n_features, rng),
        "answer"     : None,
        "samples"    : len(y),
        "depth"      : depth,
    }


# ── PREDICT with a single tree ──
def predict_tree(tree, x):
    if tree["answer"] is not None:
        return tree["answer"]
    if x[tree["feature"]] <= tree["threshold"]:
        return predict_tree(tree["left"],  x)
    else:
        return predict_tree(tree["right"], x)


# ── BUILD RANDOM FOREST ──
def build_forest(X, y, n_trees=10, max_depth=3, n_features=None, seed=42):
    trees = []
    for i in range(n_trees):
        rng = random.Random(seed + i)
        bX, by = bootstrap_sample(X, y, seed=seed + i)
        tree = build(bX, by, max_depth=max_depth, n_features=n_features, rng=rng)
        trees.append(tree)
        print(f"  Tree {i+1}/{n_trees} trained on {len(bX)} samples")
    return trees


# ── FOREST PREDICTION — majority vote ──
def predict_forest(trees, x):
    votes = [predict_tree(t, x) for t in trees]
    return max(set(votes), key=votes.count)


# ── EVALUATE accuracy on training data ──
def evaluate(trees, X, y):
    correct = sum(1 for xi, yi in zip(X, y) if predict_forest(trees, xi) == yi)
    return correct / len(y)


# ── HYPERPARAMETERS ──────────────────────────────────────────
N_TREES    = int(input("Number of trees (default 10): ").strip() or "10")
MAX_DEPTH  = int(input("Max tree depth  (default 3):  ").strip() or "3")

n_features = max(1, int(math.sqrt(len(X[0]))))  # sqrt of features — standard RF choice

print(f"\nTraining Random Forest: {N_TREES} trees, max depth {MAX_DEPTH}, "
      f"considering {n_features} feature(s) per split...\n")

# ── TRAIN ─────────────────────────────────────────────────────
forest = build_forest(X, y, n_trees=N_TREES, max_depth=MAX_DEPTH, n_features=n_features)

acc = evaluate(forest, X, y)
print(f"\nTraining accuracy: {acc * 100:.1f}%")

# ── QUICK DEMO ─────────────────────────────────────────────────
print("\nSample predictions:")
for xi, yi in zip(X, y):
    pred = predict_forest(forest, xi)
    status = "✓" if pred == yi else "✗"
    print(f"  {status} my_hp={xi[0]}, type_adv={xi[1]}, opp_hp={xi[2]} → {pred} (actual: {yi})")

# ── SAVE ──────────────────────────────────────────────────────
os.makedirs("Trained_Models", exist_ok=True)

name = input("\nName your model: ").strip() or "model"
path = "Trained_Models/" + name + ".pkl"

payload = {
    "trees"        : forest,
    "n_trees"      : N_TREES,
    "max_depth"    : MAX_DEPTH,
    "n_features"   : n_features,
    "feature_names": FEATURE_NAMES,
    "accuracy"     : acc,
}

with open(path, "wb") as f:
    pickle.dump(payload, f)

print(f"Saved to {path}")