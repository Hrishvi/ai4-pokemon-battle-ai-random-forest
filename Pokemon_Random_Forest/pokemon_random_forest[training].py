# ============================================================
# train.py — Trains a Random Forest model and saves it to disk
# ============================================================
# A Random Forest is a collection of decision trees that each
# "vote" on an answer. The most popular vote wins.
# Here we train it to decide whether to Attack or Switch in a
# Pokémon-style battle, based on HP values and type advantage.
#
# Run with:  python train.py
# ============================================================

import pickle   # Lets us save Python objects (like our trained forest) to a file
import os       # Used to create folders if they don't already exist
import random   # Used for random sampling and choosing random features
import math     # Used for the square-root formula that decides how many features to try


# ──────────────────────────────────────────────────────────────
# TRAINING DATA
# ──────────────────────────────────────────────────────────────
# This is our "experience" — examples the model learns from.
# Each row is one battle situation and the correct decision.
#
# Columns: [my_hp, type_advantage, opp_hp, correct_answer]
#   my_hp          — our Pokémon's current HP (out of 100)
#   type_advantage — 1 if we have a type advantage, 0 if we don't
#   opp_hp         — the opponent's current HP (out of 100)
#   correct_answer — what a smart trainer would do: "Attack" or "Switch"

data = [
    [80, 1, 60, "Attack"],   # We're healthy AND have type advantage → Attack
    [70, 1, 50, "Attack"],   # Same idea — favourable matchup
    [90, 1, 40, "Attack"],   # Opponent is weak and we have the edge → Attack
    [60, 0, 20, "Attack"],   # No type advantage, but opponent is nearly fainted → finish them
    [70, 0, 15, "Attack"],   # Same — opponent is very low, worth attacking anyway
    [55, 0, 80, "Switch"],   # No advantage AND opponent is healthy → not a good matchup, Switch
    [80, 0, 90, "Switch"],   # Even with decent HP, no advantage vs a strong opponent → Switch
    [20, 0, 70, "Switch"],   # We're low on HP too — dangerous to stay in → Switch
    [15, 1, 60, "Switch"],   # Type advantage won't save us at 15 HP → Switch
    [25, 0, 65, "Switch"],   # Low HP + no advantage = get out → Switch
]

# Split the data into:
#   X — the input features (what the model "sees")
#   y — the correct answers (what we want the model to learn)
X = [row[:3] for row in data]   # First 3 columns: [my_hp, type_advantage, opp_hp]
y = [row[3]  for row in data]   # Last column: "Attack" or "Switch"

# Human-readable names for each feature column (used when printing trees)
FEATURE_NAMES = ["my_hp", "type_advantage", "opp_hp"]


# ──────────────────────────────────────────────────────────────
# GINI IMPURITY
# ──────────────────────────────────────────────────────────────
# Measures how "mixed up" a group of labels is.
#   0.0  → perfectly pure   (all "Attack" OR all "Switch")
#   0.5  → maximally mixed  (50% Attack, 50% Switch)
#
# The tree always tries to split data into groups with LOW gini
# (i.e. groups where most items agree on the answer).

def gini(labels):
    total = len(labels)
    score = 1.0
    # For each unique label, subtract the square of its proportion.
    # A group with only one label will subtract 1² = 1, giving score = 0 (pure).
    for label in set(labels):
        proportion = labels.count(label) / total
        score -= proportion ** 2
    return score


# ──────────────────────────────────────────────────────────────
# BOOTSTRAP SAMPLING
# ──────────────────────────────────────────────────────────────
# Each tree in the forest is trained on a slightly different
# version of the dataset. We create these variants by randomly
# picking rows WITH REPLACEMENT — meaning the same row can be
# picked more than once, and some rows may not appear at all.
#
# This "shaking up" of the data is what makes each tree unique,
# which is why the forest gets better answers than any one tree.

def bootstrap_sample(X, y, seed=None):
    rng = random.Random(seed)   # A private random number generator (seed keeps it reproducible)
    n = len(X)
    # Pick n random row indices, allowing repeats
    indices = [rng.randint(0, n - 1) for _ in range(n)]
    # Return the rows at those indices
    return [X[i] for i in indices], [y[i] for i in indices]


# ──────────────────────────────────────────────────────────────
# FINDING THE BEST SPLIT
# ──────────────────────────────────────────────────────────────
# At each node in a decision tree we ask: "which feature and
# which threshold gives us the cleanest split?"
#
# Example split: "if my_hp <= 30 → Switch, else → Attack"
#
# Randomly choosing only a SUBSET of features to consider is
# the second key ingredient of a Random Forest — it forces
# different trees to discover different patterns.

def best_split(X, y, n_features=None, rng=None):
    if rng is None:
        rng = random

    n_total_features = len(X[0])   # How many features exist (3 in our case)

    # Default: consider sqrt(total_features) features — the standard RF rule of thumb
    if n_features is None:
        n_features = max(1, int(math.sqrt(n_total_features)))

    # Randomly pick which feature columns this split is allowed to use
    feature_indices = rng.sample(range(n_total_features), min(n_features, n_total_features))

    # Start with a placeholder "worst possible" result
    best = (None, None, 999.0)   # (feature_index, threshold, gini_score)

    for feature in feature_indices:
        # Try every unique value in this column as a possible split threshold
        for threshold in set(row[feature] for row in X):

            # Rows where the feature value is ≤ threshold go LEFT
            left  = [y[i] for i in range(len(X)) if X[i][feature] <= threshold]
            # Rows where the feature value is  > threshold go RIGHT
            right = [y[i] for i in range(len(X)) if X[i][feature] >  threshold]

            # A split is only useful if BOTH sides have at least one sample
            if not left or not right:
                continue

            # Weighted average gini: bigger groups count more
            score = (len(left) * gini(left) + len(right) * gini(right)) / len(y)

            # Keep track of whichever split produces the lowest (best) gini
            if score < best[2]:
                best = (feature, threshold, score)

    return best[0], best[1]   # Return the winning feature index and threshold


# ──────────────────────────────────────────────────────────────
# BUILDING ONE DECISION TREE
# ──────────────────────────────────────────────────────────────
# A decision tree is built by splitting data over and over until
# each group is "pure" (everyone in it agrees on the answer).
#
# The result is a nested dictionary that looks like:
#
#   Leaf node  → {"answer": "Attack", "samples": 3, "depth": 2}
#   Split node → {"feature": 0, "threshold": 30,
#                  "left": <another node>,
#                  "right": <another node>, ...}
#
# We use recursion: build() calls itself on the left and right
# sub-groups after each split.

def build(X, y, depth=0, max_depth=3, n_features=None, rng=None):
    if rng is None:
        rng = random

    # ── Base case 1: everyone agrees — this node is a leaf ──
    if len(set(y)) == 1:
        return {"answer": y[0], "samples": len(y), "depth": depth}

    # ── Base case 2: too deep or too few samples — use majority vote ──
    if depth >= max_depth or len(y) <= 1:
        majority = max(set(y), key=y.count)
        return {"answer": majority, "samples": len(y), "depth": depth}

    # Find the best question to split this group on
    feature, threshold = best_split(X, y, n_features=n_features, rng=rng)

    # ── Base case 3: no useful split found — use majority vote ──
    if feature is None:
        majority = max(set(y), key=y.count)
        return {"answer": majority, "samples": len(y), "depth": depth}

    # Divide the current rows into two groups based on the split
    left_idx  = [i for i in range(len(X)) if X[i][feature] <= threshold]
    right_idx = [i for i in range(len(X)) if X[i][feature] >  threshold]

    left_X,  left_y  = [X[i] for i in left_idx],  [y[i] for i in left_idx]
    right_X, right_y = [X[i] for i in right_idx], [y[i] for i in right_idx]

    # Recursively build the left and right branches
    return {
        "feature"     : feature,                    # Which column to split on (0, 1, or 2)
        "feature_name": FEATURE_NAMES[feature],     # Human-readable name for printing
        "threshold"   : threshold,                  # The value we compare against
        "left"        : build(left_X,  left_y,  depth + 1, max_depth, n_features, rng),
        "right"       : build(right_X, right_y, depth + 1, max_depth, n_features, rng),
        "answer"      : None,                       # None means "this is not a leaf yet"
        "samples"     : len(y),                     # How many training rows reached this node
        "depth"       : depth,                      # How far down the tree we are
    }


# ──────────────────────────────────────────────────────────────
# PREDICTING WITH ONE TREE
# ──────────────────────────────────────────────────────────────
# Walk down the tree by answering each split question until
# we reach a leaf, then return that leaf's answer.

def predict_tree(tree, x):
    # If this node has an answer stored, we've reached a leaf → return it
    if tree["answer"] is not None:
        return tree["answer"]

    # Otherwise, compare the input feature to the split threshold
    if x[tree["feature"]] <= tree["threshold"]:
        return predict_tree(tree["left"],  x)   # Go left if ≤ threshold
    else:
        return predict_tree(tree["right"], x)   # Go right if > threshold


# ──────────────────────────────────────────────────────────────
# BUILDING THE RANDOM FOREST
# ──────────────────────────────────────────────────────────────
# Trains many trees, each on a slightly different bootstrap
# sample of the data, using a different random seed so they
# each pick different feature subsets at each split.

def build_forest(X, y, n_trees=10, max_depth=3, n_features=None, seed=42):
    trees = []
    for i in range(n_trees):
        rng = random.Random(seed + i)              # Each tree gets its own random generator
        bX, by = bootstrap_sample(X, y, seed=seed + i)   # A unique scrambled dataset
        tree = build(bX, by, max_depth=max_depth, n_features=n_features, rng=rng)
        trees.append(tree)
        print(f"  Tree {i+1}/{n_trees} trained on {len(bX)} samples")
    return trees   # Return the list of all trained trees


# ──────────────────────────────────────────────────────────────
# FOREST PREDICTION — MAJORITY VOTE
# ──────────────────────────────────────────────────────────────
# Every tree votes on the answer. Whichever answer gets the
# most votes is returned as the forest's final prediction.

def predict_forest(trees, x):
    votes = [predict_tree(t, x) for t in trees]        # Collect one vote per tree
    return max(set(votes), key=votes.count)             # Return the most common vote


# ──────────────────────────────────────────────────────────────
# EVALUATE ACCURACY
# ──────────────────────────────────────────────────────────────
# Runs every training example through the forest and counts
# how often the prediction matches the correct answer.
# (Training accuracy can be misleadingly high — the model has
#  "seen" these examples, so it's partly memorising them.)

def evaluate(trees, X, y):
    correct = sum(1 for xi, yi in zip(X, y) if predict_forest(trees, xi) == yi)
    return correct / len(y)   # Fraction correct, e.g. 0.9 = 90%


# ──────────────────────────────────────────────────────────────
# STEP 1 — ASK THE USER FOR HYPERPARAMETERS
# ──────────────────────────────────────────────────────────────
# Hyperparameters are settings you choose BEFORE training begins.
# They control the forest's size and complexity.

N_TREES   = int(input("Number of trees (default 10): ").strip() or "10")
MAX_DEPTH = int(input("Max tree depth  (default 3):  ").strip() or "3")

# How many features to consider at each split.
# sqrt(number_of_features) is the standard Random Forest default.
n_features = max(1, int(math.sqrt(len(X[0]))))

print(f"\nTraining Random Forest: {N_TREES} trees, max depth {MAX_DEPTH}, "
      f"considering {n_features} feature(s) per split...\n")


# ──────────────────────────────────────────────────────────────
# STEP 2 — TRAIN
# ──────────────────────────────────────────────────────────────

forest = build_forest(X, y, n_trees=N_TREES, max_depth=MAX_DEPTH, n_features=n_features)

acc = evaluate(forest, X, y)
print(f"\nTraining accuracy: {acc * 100:.1f}%")


# ──────────────────────────────────────────────────────────────
# STEP 3 — QUICK SANITY CHECK
# ──────────────────────────────────────────────────────────────
# Print each training example alongside its prediction so we
# can visually spot any mistakes (✗ means the model got it wrong).

print("\nSample predictions:")
for xi, yi in zip(X, y):
    pred   = predict_forest(forest, xi)
    status = "✓" if pred == yi else "✗"
    print(f"  {status} my_hp={xi[0]}, type_adv={xi[1]}, opp_hp={xi[2]} → {pred} (actual: {yi})")


# ──────────────────────────────────────────────────────────────
# STEP 4 — SAVE THE TRAINED MODEL
# ──────────────────────────────────────────────────────────────
# We use Python's "pickle" module to serialise (convert to bytes)
# the entire forest and write it to a .pkl file.
# predict.py will later unpickle the same file to reuse the model.

os.makedirs("Trained_Models", exist_ok=True)   # Create the folder if it doesn't exist

name = input("\nName your model: ").strip() or "model"
path = "Trained_Models/" + name + ".pkl"

# Bundle everything the loader will need into one dictionary
payload = {
    "trees"        : forest,        # The list of trained decision trees
    "n_trees"      : N_TREES,       # Saved for display purposes
    "max_depth"    : MAX_DEPTH,     # Saved for display purposes
    "n_features"   : n_features,    # How many features were considered per split
    "feature_names": FEATURE_NAMES, # Column labels (helpful when reading the tree structure)
    "accuracy"     : acc,           # Training accuracy (shown when the model is loaded)
}

# "wb" = write binary — pickle files must be opened in binary mode
with open(path, "wb") as f:
    pickle.dump(payload, f)

print(f"Saved to {path}")
