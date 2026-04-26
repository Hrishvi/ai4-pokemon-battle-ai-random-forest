# ============================================================
# predict.py — Loads a saved Random Forest and makes predictions
# ============================================================
# After you've trained a model with train.py, run this script
# to interactively test it. You'll enter three battle stats
# and the forest will vote on whether to "Attack" or "Switch".
#
# Run with:  python predict.py
# ============================================================

import pickle   # Lets us load the model that train.py saved to disk
import os       # Used to list files in the Trained_Models folder


# ──────────────────────────────────────────────────────────────
# PREDICTING WITH ONE TREE
# ──────────────────────────────────────────────────────────────
# A decision tree is a series of yes/no questions stored as a
# nested dictionary. We "walk" the tree by answering each
# question until we reach a leaf node that holds the answer.
#
# Example path through the tree:
#   "Is my_hp ≤ 30?" → Yes → "Is opp_hp ≤ 50?" → No → "Switch"

def predict_tree(tree, x):
    # A leaf node always has a non-None "answer" key.
    # When we reach one, the journey is over — return the decision.
    if tree["answer"] is not None:
        return tree["answer"]

    # This is an internal (split) node. Compare the input value
    # for this feature against the stored threshold.
    if x[tree["feature"]] <= tree["threshold"]:
        return predict_tree(tree["left"],  x)   # ≤ threshold → follow the left branch
    else:
        return predict_tree(tree["right"], x)   # > threshold → follow the right branch


# ──────────────────────────────────────────────────────────────
# FOREST PREDICTION — MAJORITY VOTE WITH CONFIDENCE
# ──────────────────────────────────────────────────────────────
# Every tree in the forest independently predicts an answer.
# Whichever answer appears most often wins (majority vote).
# Confidence = what fraction of trees agreed on that answer.
#
# Example: 8 out of 10 trees say "Attack"
#   → label = "Attack", confidence = 80%

def predict_forest(trees, x):
    votes = [predict_tree(t, x) for t in trees]   # One vote from each tree

    # Find the label that appears most often in the votes list
    label = max(set(votes), key=votes.count)

    # Confidence: fraction of trees that voted for the winning label
    confidence = votes.count(label) / len(votes)

    return label, confidence   # e.g. ("Attack", 0.8)


# ──────────────────────────────────────────────────────────────
# STEP 1 — FIND AND LOAD A SAVED MODEL
# ──────────────────────────────────────────────────────────────
# train.py saved the model as a .pkl file using Python's pickle
# module. Here we find all .pkl files and ask the user to pick one.

# List every file in the Trained_Models folder that ends with ".pkl"
models = [f for f in os.listdir("Trained_Models") if f.endswith(".pkl")]

# If the folder is empty there's nothing to load — tell the user and stop
if not models:
    print("No models found in Trained_Models/. Run train.py first.")
    exit()

print("Available models:", ", ".join(models))

# Let the user type a model name (without the .pkl extension).
# If they just press Enter, we default to the first model found.
name = input("Model to load (without .pkl): ").strip() or models[0].replace(".pkl", "")
path = f"Trained_Models/{name}.pkl"

# "rb" = read binary — pickle files must be opened in binary mode
with open(path, "rb") as f:
    payload = pickle.load(f)   # Reconstruct the Python dictionary that was saved

# Unpack the pieces we need from the saved payload
trees         = payload["trees"]                                    # The list of decision trees
feature_names = payload.get("feature_names", ["my_hp", "type_advantage", "opp_hp"])
n_trees       = payload.get("n_trees", len(trees))                 # Number of trees (for display)

print(f"\nLoaded '{name}' — {n_trees} trees, "
      f"training accuracy: {payload.get('accuracy', 0)*100:.1f}%\n")


# ──────────────────────────────────────────────────────────────
# HELPER — SAFE INPUT WITH VALIDATION
# ──────────────────────────────────────────────────────────────
# Keeps asking until the user enters a valid number.
# Supports an optional whitelist of allowed values (e.g. only 0 or 1).
# Typing "q" at any prompt quits gracefully.

def ask(prompt, lo, hi, allowed=None):
    while True:
        raw = input(prompt).strip()

        # "q" is the quit signal — return None so the caller can break out
        if raw.lower() == "q":
            return None

        try:
            val = float(raw)   # Convert the typed text to a number

            # If there's a whitelist, reject values not in it
            if allowed is not None and val not in allowed:
                print(f"  Please enter one of: {', '.join(str(a) for a in allowed)}\n")
                continue

            # Reject values outside the allowed range
            if not (lo <= val <= hi):
                print(f"  Please enter a value between {lo} and {hi}.\n")
                continue

            return val   # All checks passed — return the validated number

        except ValueError:
            # float() raised an error because the input wasn't a number
            print("  Invalid input — please enter a number.\n")


# ──────────────────────────────────────────────────────────────
# STEP 2 — INTERACTIVE PREDICTION LOOP
# ──────────────────────────────────────────────────────────────
# Ask the user for three battle stats, feed them to the forest,
# and print the decision along with how confident the forest is.
# We loop up to 3 times; typing "q" exits early.

print("Enter 'q' at any prompt to quit.\n")

for round_num in range(1, 4):   # Rounds 1, 2, 3
    print(f"── Prediction {round_num} of 3 ──")

    # ── Collect the three input features ──────────────────────

    # Our Pokémon's current HP (1–100)
    my_hp = ask("  My HP        (1–100): ", 1, 100)
    if my_hp is None:
        break   # User typed "q" → stop the loop

    # Whether we have a type advantage: 1 = yes, 0 = no
    type_adv = ask("  Type advantage (0/1): ", 0, 1, allowed=[0, 1])
    if type_adv is None:
        break

    # Opponent's current HP (1–100)
    opp_hp = ask("  Opponent HP  (1–100): ", 1, 100)
    if opp_hp is None:
        break

    # Pack the three values into a list in the same order the model expects:
    # [my_hp, type_advantage, opp_hp]
    x = [my_hp, type_adv, opp_hp]

    # ── Ask the forest for its decision ───────────────────────
    label, conf = predict_forest(trees, x)

    # Print the result: decision and what percentage of trees agreed
    print(f"\n  → {label}  (confidence: {conf*100:.0f}% of {n_trees} trees)\n")

print("Done!")
