# Neural Network Architecture Guide
## Water Potability Binary Classification

---

## 1. Problem Definition

| Item | Detail |
|---|---|
| Task | Binary Classification |
| Input | 9 numeric water quality features |
| Output | `0` = Not Potable, `1` = Potable |
| Dataset size | 3,276 samples |
| Class ratio | 61% Not Potable / 39% Potable (mild imbalance) |
| Model type | Multilayer Perceptron (MLP) — Feedforward Neural Network |

---

## 2. Data Pipeline

Before the data even reaches the network, it must be properly prepared.

```
Raw Data (water_potability_master_clean.xlsx)
   │
   ▼
[1] Handle Missing Values
   │    → Already done: imputed using mean/median per class
   ▼
[2] Remove Duplicates
   │    → Already done: 0 duplicates confirmed
   ▼
[3] Feature Scaling  ←  CRITICAL for Neural Networks
   │    → StandardScaler: x' = (x − μ) / σ
   │    → Ensures all features contribute equally to gradient updates
   │    → Use: water_potability_scaled.xlsx
   ▼
[4] Handle Class Imbalance
   │    → Option A: SMOTE — synthetically generate minority class samples
   │    → Option B: class_weight in loss — penalize misclassifying class 1 more
   │    → Recommended: class_weight = {0: 1.0, 1: 1.56}
   ▼
[5] Train / Validation / Test Split
        70%  (2,293)    15%  (491)    15%  (492)
       Training         Validation      Test
     (fit weights)   (tune, monitor)  (final eval)
```

### Why Scaling is Critical for Neural Networks

Without scaling, features with large ranges (e.g., `Solids` in ppm) will dominate gradient updates during backpropagation, causing:
- Slow or unstable convergence
- Neurons saturating (gradients vanishing)
- Poor generalization

---

## 3. Features and Target

### Input Features (X) — 9 neurons in input layer

| # | Feature | Description | Unit |
|---|---|---|---|
| 1 | `ph` | Acidity/alkalinity | 0–14 |
| 2 | `Hardness` | Calcium and magnesium content | mg/L |
| 3 | `Solids` | Total dissolved solids | ppm |
| 4 | `Chloramines` | Chloramines concentration | ppm |
| 5 | `Sulfate` | Sulfate dissolved amount | mg/L |
| 6 | `Conductivity` | Electrical conductivity | μS/cm |
| 7 | `Organic_carbon` | Organic carbon amount | ppm |
| 8 | `Trihalomethanes` | Trihalomethane concentration | μg/L |
| 9 | `Turbidity` | Water clarity (light scattering) | NTU |

### Target (y) — 1 neuron in output layer

```
Potability:  0 → Not Potable  (1,998 samples — 61%)
             1 → Potable      (1,278 samples — 39%)
```

---

## 4. Neural Network Architecture

### Full Layer-by-Layer Diagram

```
                    ┌─────────────────────────────────┐
  INPUT LAYER       │  x₁  x₂  x₃  x₄  x₅  x₆  x₇  x₈  x₉  │   9 neurons
  (9 features)      └──────────────┬──────────────────┘
                                   │  Fully Connected (Dense)
                                   │  81 weights + 64 biases
                    ┌──────────────▼──────────────────┐
  HIDDEN LAYER 1    │   64 neurons  →  ReLU activation │
                    │   + Batch Normalization           │
                    │   + Dropout (rate = 0.3)          │
                    └──────────────┬──────────────────┘
                                   │  Fully Connected (Dense)
                                   │  64×32 = 2,048 weights + 32 biases
                    ┌──────────────▼──────────────────┐
  HIDDEN LAYER 2    │   32 neurons  →  ReLU activation │
                    │   + Batch Normalization           │
                    │   + Dropout (rate = 0.3)          │
                    └──────────────┬──────────────────┘
                                   │  Fully Connected (Dense)
                                   │  32×16 = 512 weights + 16 biases
                    ┌──────────────▼──────────────────┐
  HIDDEN LAYER 3    │   16 neurons  →  ReLU activation │
                    └──────────────┬──────────────────┘
                                   │  Fully Connected (Dense)
                                   │  16×1 = 16 weights + 1 bias
                    ┌──────────────▼──────────────────┐
  OUTPUT LAYER      │   1 neuron   →  Sigmoid activation│
                    └──────────────┬──────────────────┘
                                   │
                           P(Potable) ∈ [0, 1]
                                   │
                           Threshold @ 0.5
                                   │
                    ┌──────────────▼──────────────────┐
                    │   0 = Not Potable                │
                    │   1 = Potable                    │
                    └─────────────────────────────────┘
```

### Parameter Count Summary

| Layer | Input → Output | Weights | Biases | Total Params |
|---|---|---|---|---|
| Dense 1 | 9 → 64 | 576 | 64 | 640 |
| Dense 2 | 64 → 32 | 2,048 | 32 | 2,080 |
| Dense 3 | 32 → 16 | 512 | 16 | 528 |
| Dense 4 | 16 → 1 | 16 | 1 | 17 |
| **Total** | | | | **3,265** |

This is a lightweight model — appropriate for a 3,276-sample dataset to avoid overfitting.

---

## 5. Activation Functions

### Hidden Layers — ReLU (Rectified Linear Unit)

$$
f(z) = \max(0, z)
$$

```
Output
  │
  │          /
  │         /
  │        /
  │       /
  ─────── ─────────── z
       0
```

**Why ReLU for hidden layers:**
- Computationally efficient (simple threshold)
- Does not saturate for positive values → gradients flow well
- Introduces non-linearity, allowing the network to learn complex decision boundaries
- Default choice for hidden layers in modern MLPs

### Output Layer — Sigmoid

$$
\sigma(z) = \frac{1}{1 + e^{-z}}
$$

```
Output
1.0  ──────────────────╮
                        │
0.5  ──────────────╮    │
                   │    │
0.0  ─────────────╯─────── z
              0
```

**Why Sigmoid for output:**
- Squashes any value to range [0, 1]
- Directly interpretable as a probability: P(Potable = 1)
- Pairs naturally with Binary Cross-Entropy loss

---

## 6. Forward Pass (How a Prediction is Made)

For each sample, the network computes:

**Layer computation:**
$$
\mathbf{a}^{(l)} = f\left(\mathbf{W}^{(l)} \cdot \mathbf{a}^{(l-1)} + \mathbf{b}^{(l)}\right)
$$

Where:
- $\mathbf{W}^{(l)}$ = weight matrix of layer $l$
- $\mathbf{b}^{(l)}$ = bias vector of layer $l$
- $f$ = activation function (ReLU or Sigmoid)
- $\mathbf{a}^{(l)}$ = output (activations) of layer $l$

**Step-by-step for one sample:**

```
x = [ph, Hardness, Solids, Chloramines, Sulfate,
     Conductivity, Organic_carbon, Trihalomethanes, Turbidity]

→ z₁ = W₁·x + b₁          (9 → 64 linear transform)
→ a₁ = ReLU(z₁)            (64 activations)
→ a₁ = Dropout(a₁)         (randomly zero 30% during training)

→ z₂ = W₂·a₁ + b₂          (64 → 32 linear transform)
→ a₂ = ReLU(z₂)            (32 activations)
→ a₂ = Dropout(a₂)         (randomly zero 30% during training)

→ z₃ = W₃·a₂ + b₃          (32 → 16 linear transform)
→ a₃ = ReLU(z₃)            (16 activations)

→ z₄ = W₄·a₃ + b₄          (16 → 1 linear transform)
→ ŷ  = Sigmoid(z₄)         (scalar probability ∈ [0,1])

→ prediction = 1 if ŷ ≥ 0.5 else 0
```

---

## 7. Loss Function — Binary Cross-Entropy

The network learns by minimizing the **Binary Cross-Entropy** loss:

$$
\mathcal{L} = -\frac{1}{N} \sum_{i=1}^{N} \left[ y_i \log(\hat{y}_i) + (1 - y_i) \log(1 - \hat{y}_i) \right]
$$

Where:
- $y_i \in \{0, 1\}$ = true label
- $\hat{y}_i \in (0, 1)$ = predicted probability from Sigmoid
- $N$ = batch size

**Intuition:**
- When $y=1$ and $\hat{y} \to 1$: loss $\to 0$ (correct, confident)
- When $y=1$ and $\hat{y} \to 0$: loss $\to \infty$ (wrong, confident — heavily penalized)
- The loss pushes the network to output probabilities close to the true labels

---

## 8. Backpropagation and Weight Updates

The network learns by computing gradients of the loss with respect to every weight using the **chain rule**, then updating weights via **gradient descent**.

```
Forward Pass
─────────────────────────────────────────▶
x → Layer1 → Layer2 → Layer3 → Output → Loss L

Backward Pass (Backpropagation)
◀─────────────────────────────────────────
∂L/∂W₄ ← ∂L/∂W₃ ← ∂L/∂W₂ ← ∂L/∂W₁
```

### Optimizer — Adam (Adaptive Moment Estimation)

Adam is the recommended optimizer. It combines:
- **Momentum** — accumulates past gradients to smooth updates
- **RMSProp** — adapts learning rate per parameter

Update rule:
$$
W \leftarrow W - \frac{\eta}{\sqrt{\hat{v}} + \epsilon} \cdot \hat{m}
$$

Where $\hat{m}$ = bias-corrected first moment, $\hat{v}$ = bias-corrected second moment.

**Adam hyperparameters:**

| Parameter | Value | Description |
|---|---|---|
| `learning_rate` | `0.001` | Step size for weight updates |
| `beta_1` | `0.9` | Momentum decay (default) |
| `beta_2` | `0.999` | RMSProp decay (default) |
| `epsilon` | `1e-7` | Numerical stability term |

---

## 9. Regularization Techniques

Without regularization, a neural network will memorize the training data (overfit) and fail to generalize.

### Dropout

During each training step, randomly sets a fraction of neuron outputs to **zero**.

```
Normal forward pass:          With Dropout (rate=0.3):
[a₁, a₂, a₃, a₄, a₅]   →   [a₁, 0, a₃, 0, a₅]
                                      ↑   ↑
                               randomly dropped
```

- Forces the network to not rely on any single neuron
- Acts as training an ensemble of many sub-networks
- Applied only during **training**, not during inference

### Batch Normalization

Normalizes the output of each layer before the activation function:

$$
\hat{z} = \frac{z - \mu_{\text{batch}}}{\sqrt{\sigma^2_{\text{batch}} + \epsilon}}
$$

**Benefits:**
- Stabilizes and accelerates training
- Reduces sensitivity to weight initialization
- Acts as a mild regularizer

### Early Stopping

Monitor validation loss during training. Stop when it stops improving.

```
Epoch  Train Loss   Val Loss
─────────────────────────────
  10     0.612       0.621
  20     0.571       0.574
  30     0.543       0.548
  40     0.521       0.535
  50     0.498       0.531   ← val loss starts rising
  60     0.473       0.548   ← STOP here, restore epoch 40 weights
```

**Patience = 10–15 epochs** is a common setting.

---

## 10. Training Configuration

| Setting | Value | Reason |
|---|---|---|
| Loss function | `binary_crossentropy` | Standard for binary output |
| Optimizer | `Adam` | Adaptive, fast convergence |
| Learning rate | `0.001` | Standard Adam default |
| Batch size | `32` | Good balance of speed and gradient quality |
| Epochs | `100` (with early stopping) | Cap training time |
| Early stopping patience | `15` | Restore best weights |
| Hidden activation | `ReLU` | Fast, avoids vanishing gradients |
| Output activation | `Sigmoid` | Outputs probability |
| Dropout rate | `0.3` | 30% neurons dropped per step |
| Batch Normalization | Yes | Stabilizes training |
| Weight initialization | `Glorot Uniform` (default) | Balanced variance across layers |
| Class weight | `{0: 1.0, 1: 1.56}` | Compensates for imbalance |

---

## 11. Decision Boundary

```
                Not Potable (0)  |  Potable (1)
                                 │
   ────────────────────────────[σ(z) = 0.5]──────────────
                                 │  z = 0
         σ(z) < 0.5             │        σ(z) ≥ 0.5
         → Predict 0            │        → Predict 1
```

The threshold of **0.5** can be tuned:
- **Lower threshold (e.g., 0.3):** Predict more samples as Potable → higher Recall, lower Precision
- **Higher threshold (e.g., 0.7):** Predict fewer samples as Potable → higher Precision, lower Recall

> For this problem, a **lower threshold** is preferable — it's safer to over-predict potability than to miss truly safe water.

---

## 12. Evaluation Metrics

Because the dataset has mild class imbalance, **accuracy alone is not sufficient**.

### Confusion Matrix

```
                    Predicted
                  │  Not Potable (0) │  Potable (1)
──────────────────┼──────────────────┼──────────────
Actual  Not (0)   │       TN         │     FP
        Potable(1)│       FN         │     TP
```

### Metrics

| Metric | Formula | Interpretation |
|---|---|---|
| Accuracy | $(TP+TN) / N$ | Overall correct predictions |
| Precision | $TP / (TP+FP)$ | Of predicted potable, how many truly are |
| Recall | $TP / (TP+FN)$ | Of all truly potable, how many we caught |
| F1-Score | $2 \times (P \times R)/(P+R)$ | Harmonic mean of Precision & Recall |
| ROC-AUC | Area under ROC curve | Discrimination power across all thresholds |

> **Priority metric: Recall for class 1** — predicting unsafe water as safe is more dangerous than the reverse.

### Training Curves to Monitor

```
Loss vs Epochs:                Accuracy vs Epochs:
                               
Loss                           Accuracy
│ ╲                            │         ╱────
│  ╲  (train)                  │        ╱  (val)
│   ╲──────────                │  ──────
│        ╲──── (val)           │ (train)
└────────────── Epoch          └────────────── Epoch

 Good: val loss decreasing       Good: val acc increasing
 Bad:  val loss rising while      Bad: val acc plateauing while
       train loss still drops           train acc keeps rising
       → OVERFITTING                    → OVERFITTING
```

---

## 13. Full Code Implementation (Keras / TensorFlow)

```python
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, roc_auc_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt

# ── 1. Load Data ──────────────────────────────────────────
df = pd.read_excel("water_potability_scaled.xlsx")
X = df.drop("Potability", axis=1).values
y = df["Potability"].values

# ── 2. Train / Val / Test Split ───────────────────────────
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.30, stratify=y, random_state=42
)
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=42
)

print(f"Train: {X_train.shape} | Val: {X_val.shape} | Test: {X_test.shape}")

# ── 3. Class Weights ──────────────────────────────────────
neg, pos = np.bincount(y_train)
class_weight = {0: 1.0, 1: neg / pos}
print(f"Class weight for Potable (1): {neg/pos:.2f}")

# ── 4. Build Model ────────────────────────────────────────
model = Sequential([
    # Input → Hidden Layer 1
    Dense(64, input_shape=(9,), kernel_initializer='glorot_uniform'),
    BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    Dropout(0.3),

    # Hidden Layer 2
    Dense(32, kernel_initializer='glorot_uniform'),
    BatchNormalization(),
    tf.keras.layers.Activation('relu'),
    Dropout(0.3),

    # Hidden Layer 3
    Dense(16, activation='relu'),

    # Output Layer
    Dense(1, activation='sigmoid')
])

model.summary()

# ── 5. Compile ────────────────────────────────────────────
model.compile(
    optimizer=Adam(learning_rate=0.001),
    loss='binary_crossentropy',
    metrics=['accuracy',
             tf.keras.metrics.Precision(name='precision'),
             tf.keras.metrics.Recall(name='recall'),
             tf.keras.metrics.AUC(name='auc')]
)

# ── 6. Callbacks ──────────────────────────────────────────
early_stop = EarlyStopping(
    monitor='val_loss',
    patience=15,
    restore_best_weights=True,
    verbose=1
)

# ── 7. Train ──────────────────────────────────────────────
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=100,
    batch_size=32,
    class_weight=class_weight,
    callbacks=[early_stop],
    verbose=1
)

# ── 8. Evaluate on Test Set ───────────────────────────────
y_pred_proba = model.predict(X_test).flatten()
y_pred = (y_pred_proba >= 0.5).astype(int)

print("\n── Classification Report ──")
print(classification_report(y_test, y_pred,
      target_names=["Not Potable", "Potable"]))
print(f"ROC-AUC:  {roc_auc_score(y_test, y_pred_proba):.4f}")
print(f"\nConfusion Matrix:\n{confusion_matrix(y_test, y_pred)}")

# ── 9. Plot Training Curves ───────────────────────────────
fig, axes = plt.subplots(1, 2, figsize=(12, 4))

axes[0].plot(history.history['loss'], label='Train Loss')
axes[0].plot(history.history['val_loss'], label='Val Loss')
axes[0].set_title('Loss vs Epochs')
axes[0].set_xlabel('Epoch')
axes[0].set_ylabel('Loss')
axes[0].legend()

axes[1].plot(history.history['accuracy'], label='Train Accuracy')
axes[1].plot(history.history['val_accuracy'], label='Val Accuracy')
axes[1].set_title('Accuracy vs Epochs')
axes[1].set_xlabel('Epoch')
axes[1].set_ylabel('Accuracy')
axes[1].legend()

plt.tight_layout()
plt.savefig("training_curves.png", dpi=150)
plt.show()
```

---

## 14. Expected Results (Reference)

| Metric | Expected Range |
|---|---|
| Test Accuracy | 68–74% |
| ROC-AUC | 0.72–0.80 |
| Recall (Potable) | 0.60–0.75 |
| F1-Score (Potable) | 0.60–0.72 |

> Water potability is an inherently difficult classification problem due to overlapping feature distributions between classes. These ranges are typical for MLP on this dataset.

---

*Guide prepared for IS2 — Water Potability Binary Classification Project*
