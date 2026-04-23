# 🎮 Pokémon Battle AI — Random Forest from Scratch

> A lightweight Random Forest classifier built **without any ML libraries** that predicts the optimal Pokémon battle move — **Attack** or **Switch** — using HP values and type advantage.

![Python](https://img.shields.io/badge/Python-3.8%2B-blue?logo=python&logoColor=white)
![ML from Scratch](https://img.shields.io/badge/ML-From%20Scratch-orange)
![License](https://img.shields.io/badge/License-MIT-green)
![No Dependencies](https://img.shields.io/badge/Dependencies-None-brightgreen)

---

## 🧠 What Is This?

This project implements a **Random Forest decision tree classifier from scratch** in pure Python — no scikit-learn, no pandas, no NumPy. It's trained on Pokémon battle scenarios and learns when to attack versus switch out based on three simple features.

It's an ideal resource for anyone learning:
- How **decision trees** split data using the Gini impurity criterion
- How **bootstrap sampling** and feature randomness create a forest
- How **majority voting** aggregates tree predictions into a final answer
- How to **serialize and reload** a trained ML model using `pickle`

---

## ✨ Features

- ✅ Random Forest built entirely from scratch in Python
- ✅ Gini impurity-based splitting for optimal decision boundaries
- ✅ Bootstrap sampling (bagging) per tree for variance reduction
- ✅ Random feature subsets per split (the core RF trick)
- ✅ Majority vote with **confidence score** at prediction time
- ✅ Save and load trained models via `.pkl` files
- ✅ Interactive CLI for both training and prediction
- ✅ Zero external ML dependencies

---

## 📁 Project Structure

```
pokemon-battle-ai-random-forest/
│
├── train.py                  # Train a new Random Forest and save it
├── predict.py                # Load a saved model and make predictions
├── Trained_Models/
│   └── pokemon_random_forest.pkl   # Pre-trained model (10 trees, depth 3)
└── README.md
```

---

## 🚀 Quick Start

**1. Clone the repository**
```bash
git clone https://github.com/your-username/pokemon-battle-ai-random-forest.git
cd pokemon-battle-ai-random-forest
```

**2. Train a new model**
```bash
python train.py
```
You'll be prompted for the number of trees and max depth. The model is saved to `Trained_Models/`.

**3. Make predictions**
```bash
python predict.py
```
Enter your Pokémon's HP, your type advantage, and the opponent's HP — the forest votes on whether to **Attack** or **Switch**.

---

## 🎯 Input Features

| Feature | Description | Range |
|---|---|---|
| `my_hp` | Your Pokémon's current HP | 1 – 100 |
| `type_advantage` | Do you have a type advantage? | 0 (No) / 1 (Yes) |
| `opp_hp` | Opponent's current HP | 1 – 100 |

**Output:** `Attack` or `Switch` with a confidence percentage (e.g. *"Attack — 80% of 10 trees"*)

---

## 🌲 How the Random Forest Works

```
Training
  └── For each tree (1 to N):
        ├── Bootstrap sample the training data (sample with replacement)
        ├── At each node, consider √features random features
        ├── Pick the split that minimises Gini impurity
        └── Recurse until pure leaves or max depth reached

Prediction
  └── Each tree votes independently
        └── Majority vote → final label + confidence score
```

---

## 🗂️ Training Data

The model is trained on 10 hand-crafted battle scenarios covering clear attack and switch situations:

```python
[my_hp, type_advantage, opp_hp] → decision

[80, 1, 60] → Attack   # healthy + type advantage
[15, 1, 60] → Switch   # too low HP even with advantage
[55, 0, 80] → Switch   # no advantage, opponent healthy
[60, 0, 20] → Attack   # opponent nearly fainted
```

---

## ⚙️ Hyperparameters

| Parameter | Default | Effect |
|---|---|---|
| `N_TREES` | 10 | More trees = higher stability, slower training |
| `MAX_DEPTH` | 3 | Deeper trees = more complex rules, risk of overfitting |
| `n_features` | √3 ≈ 2 | Features considered per split (standard RF heuristic) |

---

## 📊 Model Performance

The pre-trained model (`pokemon_random_forest.pkl`) achieves **100% training accuracy** on the 10-sample dataset with 10 trees at max depth 3.

---

## 💡 Learning Outcomes

This project is a great hands-on reference for understanding:

- **Gini Impurity** — how decision trees measure node purity
- **Bagging** — why training each tree on a different sample reduces overfitting
- **Feature Randomness** — why considering only √n features per split decorrelates trees
- **Ensemble Methods** — why many weak learners beat one strong learner
- **Model Persistence** — how to save and reload ML models in Python

---

## 🛠️ Requirements

- Python 3.8+
- Standard library only (`pickle`, `random`, `math`, `os`)

---

## 📄 License

MIT License — free to use, modify, and distribute.

---

## 🔖 Tags

`machine-learning` `random-forest` `decision-tree` `python` `pokemon` `from-scratch` `gini-impurity` `bagging` `ensemble-learning` `no-libraries` `classification` `game-ai` `tutorial`
