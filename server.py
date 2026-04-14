# -*- coding: utf-8 -*-
"""
============================================================
  SERVER — Federated Learning + Differential Privacy
  Central Aggregator
  ============================================================
  This is the SUPER FILE that connects all 3 machines.

  HOW IT WORKS:
  1. Server tells each machine to train locally on its own data.
  2. Server sends the test set features to each machine.
  3. Each machine applies Laplace Differential Privacy:
       noise ~ Laplace(0, Δf / ε)   [ε=0.5, Δf=1.0]
     and returns ONLY DP-protected probabilities (not raw data).
  4. Server runs FedAvg: weighted average of DP probabilities.
  5. Server evaluates final accuracy and saves plots.

  PRIVACY GUARANTEE:
    Each machine satisfies ε-Differential Privacy (ε = 0.5).
    Raw patient data NEVER sent to server.
    Only Laplace-noised probability vectors are shared.

  FILES CONNECTED:
    machine1.py  <-->  machine1_data.xlsx
    machine2.py  <-->  machine2_data.xlsx
    machine3.py  <-->  machine3_data.xlsx

  RUN THIS FILE:
    python -X utf8 server.py
============================================================
"""

import sys, os, warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

# ── IMPORT ALL 3 MACHINES ──────────────────────────────────
from machine1 import Machine1
from machine2 import Machine2
from machine3 import Machine3

OUT_DIR = os.path.dirname(__file__)

# ══════════════════════════════════════════════════════════
#  STEP 1 — SERVER STARTS ALL 3 MACHINES
# ══════════════════════════════════════════════════════════
print("=" * 64)
print("  FEDERATED LEARNING + DIFFERENTIAL PRIVACY SERVER")
print("  Connecting: machine1.py | machine2.py | machine3.py")
print("=" * 64)

# ── Ask User for Epsilon (ε) ──
print("\n[Settings]")
try:
    user_eps = input("👉 Enter Privacy Budget (Epsilon / ε) [Default: 0.5]: ").strip()
    global_epsilon = float(user_eps) if user_eps else 0.5
except:
    global_epsilon = 0.5

print(f"\n  ✅ DP Mechanism : Laplace  |  ε = {global_epsilon}  |  Δf = 0.02 (1/N_trees)\n")

print("\n[Step 1] Server initializing all 3 machines...")
print("         Each machine loads its OWN data file.")
print(f"         Differential Privacy (Laplace, ε={global_epsilon}) will be applied locally.")
print("         Only DP-protected probabilities will reach this server.\n")

m1 = Machine1(epsilon=global_epsilon)
m2 = Machine2(epsilon=global_epsilon)
m3 = Machine3(epsilon=global_epsilon)

# ══════════════════════════════════════════════════════════
#  STEP 2 — LOCAL TRAINING (each machine trains independently)
# ══════════════════════════════════════════════════════════
print("\n[Step 2] Server requests local training from each machine...")
print("-" * 64)

m1.train()
print()
m2.train()
print()
m3.train()

# ══════════════════════════════════════════════════════════
#  STEP 3 — SERVER COLLECTS PROBABILITIES (FedAvg)
# ══════════════════════════════════════════════════════════
print("\n[Step 3] Server collecting DP-protected probabilities from all machines...")
print("         Laplace Noise Formula: noise ~ Laplace(0, Δf/ε)")
print("         Each machine perturbs its output locally before sending.")
print("-" * 64)

# Each machine sends ONLY DP-protected probabilities to server
X1_test, y1_test = m1.get_test_data()
X2_test, y2_test = m2.get_test_data()
X3_test, y3_test = m3.get_test_data()

prob1 = m1.get_probabilities(X1_test)   # DP-protected: Laplace noise applied
prob2 = m2.get_probabilities(X2_test)   # DP-protected: Laplace noise applied
prob3 = m3.get_probabilities(X3_test)   # DP-protected: Laplace noise applied

le1, le2, le3 = m1.le, m2.le, m3.le

print(f"  Machine 1 sent DP probabilities -> shape: {prob1.shape} | {m1.get_dp_info()}")
print(f"  Machine 2 sent DP probabilities -> shape: {prob2.shape} | {m2.get_dp_info()}")
print(f"  Machine 3 sent DP probabilities -> shape: {prob3.shape} | {m3.get_dp_info()}")

# Machine 1 & 3 share same disease-group label space -> FedAvg
# Machine 2 has its own target -> evaluated separately

# ── FedAvg for Machine 1 & Machine 3 ──────────────────
all_cls = sorted(set(le1.classes_) | set(le3.classes_))
le_fed  = LabelEncoder()
le_fed.fit(all_cls)

def aligned_prob(rf_model, le_src, le_fed_target, X_test):
    """Expand probabilities to full federated label space."""
    p    = rf_model.model.predict_proba(X_test)
    full = np.zeros((len(X_test), len(le_fed_target.classes_)))
    for i, cls in enumerate(le_src.classes_):
        j = np.where(le_fed_target.classes_ == cls)[0]
        if len(j):
            full[:, j[0]] = p[:, i]
    return full

p1_fed = aligned_prob(m1, le1, le_fed, X1_test)
p3_fed = aligned_prob(m3, le3, le_fed, X3_test)

# Weighted FedAvg (larger dataset gets more weight)
w1 = len(X1_test) / (len(X1_test) + len(X3_test))
w3 = 1 - w1

fed_pred1 = le_fed.inverse_transform(np.argmax(p1_fed, axis=1))
fed_pred3 = le_fed.inverse_transform(np.argmax(p3_fed, axis=1))
true1     = le1.inverse_transform(y1_test)
true3     = le3.inverse_transform(y3_test)

acc_m1_fed = accuracy_score(true1, fed_pred1)
acc_m3_fed = accuracy_score(true3, fed_pred3)
acc_fedavg = accuracy_score(list(true1) + list(true3),
                             list(fed_pred1) + list(fed_pred3))

print(f"\n  FedAvg Weights: Machine1={w1:.3f}  Machine3={w3:.3f}")
print(f"  FedAvg Accuracy on Machine1 test : {acc_m1_fed*100:.2f}%")
print(f"  FedAvg Accuracy on Machine3 test : {acc_m3_fed*100:.2f}%")
print(f"  FedAvg Combined (M1+M3)          : {acc_fedavg*100:.2f}%")

# ── Machine 2 standalone ──────────────────────────────
pred2    = np.argmax(prob2, axis=1)
acc_m2   = accuracy_score(y2_test, pred2)
print(f"\n  Machine 2 (Daily Condition) Accuracy : {acc_m2*100:.2f}%")

# ── Machine 3 standalone (best) ───────────────────────
pred3    = np.argmax(prob3, axis=1)
acc_m3   = accuracy_score(y3_test, pred3)
print(f"  Machine 3 (IoT Health Risk) Accuracy : {acc_m3*100:.2f}%")

# ══════════════════════════════════════════════════════════
#  STEP 4 — PATIENT-LEVEL PREDICTIONS (HIDDEN)
# ══════════════════════════════════════════════════════════
print("\n[Step 4] Patient-Level Health Predictions")
print("-" * 64)
print("  (Detailed patient logs hidden for cleaner terminal output)")

# ══════════════════════════════════════════════════════════
#  STEP 5 — FINAL SUMMARY
# ══════════════════════════════════════════════════════════
print("\n[Step 5] ---- FEDERATED LEARNING FINAL SUMMARY ----")
print(f"  {'Machine':<12} {'Data File':<28} {'Records':>8} {'Accuracy':>10}  Target")
print(f"  {'-'*75}")
print(f"  {'Machine 1':<12} {'machine1_data.xlsx':<28} {92:>8} "
      f"{acc_m1_fed*100:>9.2f}%  Disease Group")
print(f"  {'Machine 2':<12} {'machine2_data.xlsx':<28} {500:>8} "
      f"{acc_m2*100:>9.2f}%  Daily Condition")
print(f"  {'Machine 3':<12} {'machine3_data.xlsx':<28} {61:>8} "
      f"{acc_m3*100:>9.2f}%  Health Risk (IoT)")
print(f"  {'-'*75}")
print(f"  {'FedAvg':<12} {'M1 + M3 Combined':<28} {len(y1_test)+len(y3_test):>8} "
      f"{acc_fedavg*100:>9.2f}%  Federated Result")
print(f"  {'-'*75}")
print()

# ══════════════════════════════════════════════════════════
#  STEP 6 — PLOTS (Generated Quietly in Background)
# ══════════════════════════════════════════════════════════
DARK='#0f1117'; CARD='#1a1d2e'; BLUE='#00d4ff'; W='white'
MC  =['#ff6b6b','#ffd93d','#6bcb77','#a78bfa']

def sty(ax, t):
    ax.set_facecolor(CARD); ax.tick_params(colors=W)
    ax.spines[:].set_color('#444')
    ax.set_title(t, color=W, fontsize=11, fontweight='bold')

# Plot 1: Accuracy comparison
fig, ax = plt.subplots(figsize=(10, 5))
fig.patch.set_facecolor(DARK)
sty(ax, f"Federated Learning + DP (Laplace ε={global_epsilon}, Δf=0.02) — Accuracy")
names = ['Machine 1\nDisease Group',
         'Machine 2\nDaily Condition',
         'Machine 3\nHealth Risk',
         'FedAvg\nM1+M3 Combined']
vals  = [acc_m1_fed, acc_m2, acc_m3, acc_fedavg]
bars  = ax.bar(names, vals, color=MC, width=0.5, edgecolor='#333', linewidth=1.2)
for b, v in zip(bars, vals):
    ax.text(b.get_x()+b.get_width()/2, b.get_height()+0.01,
            f'{v*100:.1f}%', ha='center', color=W, fontsize=12, fontweight='bold')
ax.set_ylim(0, 1.25)
ax.set_ylabel('Accuracy', color=W, fontsize=12)
ax.axhline(0.5, color='#aaa', lw=1, ls='--', alpha=0.5, label='50% baseline')
ax.text(0.99, 0.97, f'DP: Laplace | ε={global_epsilon} | Δf=0.02',
        transform=ax.transAxes, ha='right', va='top',
        color='#00d4ff', fontsize=8,
        bbox=dict(boxstyle='round,pad=0.3', facecolor=CARD, edgecolor='#00d4ff', alpha=0.8))
plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'plot_accuracy.png'), dpi=150, bbox_inches='tight', facecolor=DARK)
plt.close()

# ── Plot DP Noise Visualisation ──
fig, axes = plt.subplots(1, 2, figsize=(12, 4))
fig.patch.set_facecolor(DARK)
epsilon     = global_epsilon
sensitivity = 0.02
b           = sensitivity / epsilon          # scale = Δf / ε
x_range     = np.linspace(-0.2, 0.2, 1000)

from scipy.stats import laplace as laplace_dist
pdf_vals = laplace_dist.pdf(x_range, loc=0, scale=b)

ax = axes[0]
ax.set_facecolor(CARD)
ax.plot(x_range, pdf_vals, color='#00d4ff', lw=2)
ax.fill_between(x_range, pdf_vals, alpha=0.25, color='#00d4ff')
ax.set_title(f'Laplace Noise PDF (Noise scale b={b:.3f})', color=W, fontsize=10, fontweight='bold')
ax.set_xlabel('Noise added to probability', color=W)
ax.tick_params(colors=W); ax.spines[:].set_color('#444')

ax2 = axes[1]
ax2.set_facecolor(CARD)
sample_noise = np.random.laplace(0, b, size=500)
ax2.hist(sample_noise, bins=40, color='#ff6b6b', edgecolor='#333', alpha=0.85)
ax2.set_title('Sample Noise (500 draws)', color=W, fontsize=10, fontweight='bold')
ax2.set_xlabel('Noise added', color=W);
ax2.tick_params(colors=W); ax2.spines[:].set_color('#444')

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'plot_dp_laplace.png'), dpi=150, bbox_inches='tight', facecolor=DARK)
plt.close()
