# predict.py — Loads a saved forest and makes predictions
# Run with: python predict.py

import pickle
import os


# ── PREDICT with a single tree ──
def predict_tree(tree, x):
    if tree["answer"] is not None:
        return tree["answer"]
    if x[tree["feature"]] <= tree["threshold"]:
        return predict_tree(tree["left"], x)
    else:
        return predict_tree(tree["right"], x)


# ── FOREST PREDICTION — majority vote with confidence ──
def predict_forest(trees, x):
    votes = [predict_tree(t, x) for t in trees]
    label = max(set(votes), key=votes.count)
    confidence = votes.count(label) / len(votes)
    return label, confidence


# ── LOAD MODEL ────────────────────────────────────────────────
models = [f for f in os.listdir("Trained_Models") if f.endswith(".pkl")]
if not models:
    print("No models found in Trained_Models/. Run train.py first.")
    exit()

print("Available models:", ", ".join(models))
name = input("Model to load (without .pkl): ").strip() or models[0].replace(".pkl", "")
path = f"Trained_Models/{name}.pkl"

with open(path, "rb") as f:
    payload = pickle.load(f)

trees         = payload["trees"]
feature_names = payload.get("feature_names", ["my_hp", "type_advantage", "opp_hp"])
n_trees       = payload.get("n_trees", len(trees))

print(f"\nLoaded '{name}' — {n_trees} trees, "
      f"training accuracy: {payload.get('accuracy', 0)*100:.1f}%\n")

# ── HELPER — prompt for a single value with validation ──
def ask(prompt, lo, hi, allowed=None):
    while True:
        raw = input(prompt).strip()
        if raw.lower() == "q":
            return None
        try:
            val = float(raw)
            if allowed is not None and val not in allowed:
                print(f"  Please enter one of: {', '.join(str(a) for a in allowed)}\n")
                continue
            if not (lo <= val <= hi):
                print(f"  Please enter a value between {lo} and {hi}.\n")
                continue
            return val
        except ValueError:
            print("  Invalid input — please enter a number.\n")


# ── INTERACTIVE PREDICTION LOOP ───────────────────────────────
print("Enter 'q' at any prompt to quit.\n")

for round_num in range(1, 4):
    print(f"── Prediction {round_num} of 3 ──")

    my_hp = ask("  My HP        (1–100): ", 1, 100)
    if my_hp is None: break

    type_adv = ask("  Type advantage (0/1): ", 0, 1, allowed=[0, 1])
    if type_adv is None: break

    opp_hp = ask("  Opponent HP  (1–100): ", 1, 100)
    if opp_hp is None: break

    x = [my_hp, type_adv, opp_hp]
    label, conf = predict_forest(trees, x)
    print(f"\n  → {label}  (confidence: {conf*100:.0f}% of {n_trees} trees)\n")

print("Done!")