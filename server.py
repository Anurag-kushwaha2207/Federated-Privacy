# -*- coding: utf-8 -*-
"""
============================================================
  SERVER — Federated Learning + Differential Privacy
  Central Aggregator
  ============================================================
  This file connects Machine 1, 2, and 3 to run a multi-round
  Horizontal Federated Learning loop using FedAvg.
  
  Each machine applies mathematically correct Laplace
  Differential Privacy (Output Perturbation) on its local weights.
============================================================
"""

import sys
import os
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import accuracy_score, confusion_matrix

np.random.seed(42)

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

# ── IMPORT CLIENTS ─────────────────────────────────────────
from machine1 import Machine1, ALPHA
from machine2 import Machine2
from machine3 import Machine3

OUT_DIR = os.path.dirname(__file__)

print("================================================================")
print("Federated Learning & Differential Privacy Aggregator")
print("Clients: Machine 1, Machine 2, Machine 3")
print("================================================================")

# ── Ask User for Epsilon (ε) or parse arguments ──
import argparse

parser = argparse.ArgumentParser(description="Federated Learning + Differential Privacy Server")
parser.add_argument("--no-dp", action="store_true", help="Run without Differential Privacy (no noise added)")
parser.add_argument("-e", "--epsilon", type=float, default=None, help="Privacy Budget (Epsilon / ε) to use directly without prompting")
parser.add_argument("--dp-clients", type=str, default="1,2,3", help="Comma-separated list of machine numbers to apply DP (e.g. 1 or 1,2)")
parser.add_argument("--dp-mode", type=str, choices=["input", "output"], default="output", help="DP Perturbation stage: 'input' (raw features) or 'output' (model weights)")
args, unknown = parser.parse_known_args()

use_dp = not args.no_dp

if not use_dp:
    global_epsilon = 1.0  # default value for machine initialization
    print("\nConfiguration:")
    print("  Differential Privacy: Disabled")
    print(f"  Regularization      : L2 (alpha = {ALPHA})\n")
    dp_enabled_clients = set()
else:
    if args.epsilon is not None:
        global_epsilon = args.epsilon
    else:
        try:
            user_eps = input("Enter Privacy Budget (Epsilon / epsilon) [default: 1.0]: ").strip()
            global_epsilon = float(user_eps) if user_eps else 1.0
        except Exception:
            global_epsilon = 1.0
            
    if global_epsilon <= 0:
        print("Error: Epsilon (privacy budget) must be strictly greater than 0.")
        sys.exit(1)

    # Parse which clients will have DP enabled
    try:
        dp_enabled_clients = {int(x.strip()) for x in args.dp_clients.split(",") if x.strip()}
    except Exception:
        dp_enabled_clients = {1, 2, 3}

    print("\nConfiguration:")
    print(f"  Differential Privacy: Enabled ({'Laplace Input Perturbation' if args.dp_mode == 'input' else 'Laplace Output Perturbation'})")
    print(f"  Privacy Budget (eps): {global_epsilon}")
    active_dp_str = ", ".join(f"Machine {x}" for x in sorted(dp_enabled_clients)) if dp_enabled_clients else "None"
    print(f"  DP Active On        : {active_dp_str}")
    print(f"  DP Mode             : {args.dp_mode.upper()} PERTURBATION")
    print(f"  Regularization      : L2 (alpha = {ALPHA})\n")

ROUNDS = 5

# Calculate epsilon per round based on sequential composition theorem:
# For Input DP, noise is applied once at feature level, so full budget is used once.
# For Output DP, noise is applied across K rounds, so total budget global_epsilon is allocated as global_epsilon / ROUNDS per round.
if use_dp:
    if args.dp_mode == "input":
        epsilon_per_round = global_epsilon
    else:
        epsilon_per_round = global_epsilon / ROUNDS
else:
    epsilon_per_round = 0.0

print("Initializing clients and partitioning dataset...")
m1 = Machine1(epsilon=epsilon_per_round if 1 in dp_enabled_clients else 0.0, dp_mode=args.dp_mode)
m2 = Machine2(epsilon=epsilon_per_round if 2 in dp_enabled_clients else 0.0, dp_mode=args.dp_mode)
m3 = Machine3(epsilon=epsilon_per_round if 3 in dp_enabled_clients else 0.0, dp_mode=args.dp_mode)

clients = [m1, m2, m3]
names = ["Machine 1", "Machine 2", "Machine 3"]

# ── STEP 2: Compute Baseline (Standalone Local Models) ──
print("\nEvaluating standalone local models (baselines)...")
print("-" * 64)

local_accuracies = []
for idx, m in enumerate(clients):
    # Train local model from scratch
    m.initialize_model()
    # Train for 100 epochs locally to achieve near-convergence for DP sensitivity mathematical guarantees
    m.local_train(epochs=100)
    
    # Evaluate locally (non-private)
    X_te, y_te = m.get_test_data()
    preds = m.model.predict(X_te)
    acc = accuracy_score(y_te, preds)
    local_accuracies.append(acc)
    print(f"  {names[idx]} train size: {m.get_train_size()} | Accuracy: {acc*100:.2f}%")

# ── STEP 3: Multi-Round Federated Learning Loop ──
print("\nStarting Federated Learning training loops (FedAvg)...")
print("-" * 64)

history_acc = []

# Re-initialize clients to start Federated Learning with fresh step counters and clean states
for m in clients:
    m.initialize_model()

# Initialize global weights to zero (from m1's structure)
global_coef, global_intercept = m1.get_weights()
global_coef.fill(0.0)
global_intercept.fill(0.0)

for r in range(1, ROUNDS + 1):
    print(f"Round {r}/{ROUNDS}:")
    
    local_weights = []
    dataset_sizes = []
    
    for idx, m in enumerate(clients):
        # 1. Server sends current global weights to client
        m.set_weights(global_coef, global_intercept)
        
        # 2. Client trains locally on its partition (5 epochs to prevent client drift)
        m.local_train(epochs=5)
        
        # 3. Client applies Output Perturbation DP if enabled for this client, else returns non-private weights
        client_num = idx + 1
        if use_dp and client_num in dp_enabled_clients:
            noisy_coef, noisy_intercept = m.get_dp_weights()
            if r == 1:
                if args.dp_mode == "input":
                    print(f"    - {names[idx]}: (Input DP Enabled, Epsilon = {global_epsilon:.4f}, Features Perturbed locally)")
                else:
                    # Calculate maximum L2 norm of the features for printing
                    R_CLIP = 2.5
                    R = min(np.max(np.linalg.norm(m.X_train, axis=1)), R_CLIP)
                    sensitivity = (2.0 * R) / (m.get_train_size() * m.alpha)
                    scale = sensitivity / epsilon_per_round
                    print(f"    - {names[idx]}: sensitivity = {sensitivity:.4f}, noise scale = {scale:.4f}, max_norm(R) = {R:.4f} (DP Enabled, Epsilon per round = {epsilon_per_round:.4f})")
        else:
            noisy_coef, noisy_intercept = m.get_weights()
            if r == 1:
                print(f"    - {names[idx]}: (DP Disabled)")
        
        local_weights.append((noisy_coef, noisy_intercept))
        dataset_sizes.append(m.get_train_size())
            
    # 4. Server aggregates weights (FedAvg)
    total_samples = sum(dataset_sizes)
    new_global_coef = np.zeros_like(global_coef)
    new_global_intercept = np.zeros_like(global_intercept)
    
    for idx in range(len(clients)):
        weight = dataset_sizes[idx] / total_samples
        c_coef, c_inter = local_weights[idx]
        new_global_coef += weight * c_coef
        new_global_intercept += weight * c_inter
        
    global_coef = new_global_coef
    global_intercept = new_global_intercept
    
    # 5. Evaluate the aggregated global model on all test sets
    all_y_true = []
    all_y_pred = []
    
    for m in clients:
        m.set_weights(global_coef, global_intercept)
        X_te, y_te = m.get_test_data()
        preds = m.model.predict(X_te)
        all_y_true.extend(y_te)
        all_y_pred.extend(preds)
        
    round_acc = accuracy_score(all_y_true, all_y_pred)
    history_acc.append(round_acc)
    print(f"    Global aggregated model accuracy: {round_acc*100:.2f}%")

# ── STEP 4: Final Summary ──
print("\n=================== FINAL SUMMARY ===================")
if use_dp:
    print(f"  Differential Privacy Status : ENABLED ({args.dp_mode.upper()} PERTURBATION)")
    if args.dp_mode == "input":
        print(f"  Total Privacy Epsilon       : {global_epsilon:.4f} (Applied at Input Layer)")
    else:
        total_composed_epsilon = epsilon_per_round * ROUNDS
        print(f"  Number of rounds (K)        : {ROUNDS}")
        print(f"  Epsilon per Round (eps_r)   : {epsilon_per_round:.4f}")
        print(f"  Total Composed Epsilon      : {total_composed_epsilon:.4f} (Sequential Composition: K * eps_r)")
else:
    print(f"  Differential Privacy Status : DISABLED")
print(f"  {'Model / Client':<18} {'Train Records':>15} {'Accuracy':>12}")
print(f"  {'-'*49}")
for idx in range(len(clients)):
    print(f"  {names[idx]:<18} {clients[idx].get_train_size():>15} {local_accuracies[idx]*100:>11.2f}%")
print(f"  {'-'*49}")
print(f"  {'FedAvg Global':<18} {total_samples:>15} {history_acc[-1]*100:>11.2f}%")
print(f"  {'-'*49}\n")

# ── STEP 5: Model Personalization (Fine-Tuning) ──
print("=================== MODEL PERSONALIZATION ===================")
print("Each client receives the global model and fine-tunes on local data...")
print("-" * 61)
personalized_accuracies = []
for idx, m in enumerate(clients):
    # Retrieve current global model accuracy on this client's test set
    m.set_weights(global_coef, global_intercept)
    X_te, y_te = m.get_test_data()
    preds_global = m.model.predict(X_te)
    acc_global = accuracy_score(y_te, preds_global)
    
    # Personalize for 10 epochs
    m.fine_tune(global_coef, global_intercept, epochs=10)
    preds_pers = m.model.predict(X_te)
    acc_pers = accuracy_score(y_te, preds_pers)
    personalized_accuracies.append(acc_pers)
    print(f"  {names[idx]} | Global Acc: {acc_global*100:.2f}% | Personalized Acc: {acc_pers*100:.2f}%")
print(f"  Average Personalized Accuracy: {np.mean(personalized_accuracies)*100:.2f}%\n")

# ── STEP 6: Risk Level Mapping & Health Alerts ──
def map_to_risk_level(pred_class):
    """Maps health event prediction class (0,1,2,3) to low/medium/high risk levels."""
    if pred_class == 0:
        return "LOW RISK"
    elif pred_class in [1, 2]:
        return "MEDIUM RISK"
    else:
        return "HIGH RISK"

def get_health_recommendation(risk_level):
    """Generates a health recommendation message based on the risk level."""
    if risk_level == "LOW RISK":
        return "Status: Normal. Recommendation: Maintain your daily exercise routine and healthy sleep patterns."
    elif risk_level == "MEDIUM RISK":
        return "Status: Mild/Moderate Event detected. Recommendation: Monitor your vital signs closely, reduce physical stress, and rest."
    else:
        return "Status: Severe Event detected! Alert: Immediate clinical consultation is advised. Avoid strenuous activities."

# Generate sample risk levels and alerts on Machine 1's local test data (using personalized model)
print("=================== SAMPLE RISK MAPPINGS & ALERTS ===================")
print("Displaying rule-based health recommendations from predicted classes...")
print("-" * 69)
class_names = ['Normal', 'Mild Event', 'Moderate Event', 'Severe Event']
X_sample, y_sample = m1.get_test_data()
preds_sample = m1.model.predict(X_sample[:4])
for i in range(len(preds_sample)):
    risk = map_to_risk_level(preds_sample[i])
    rec = get_health_recommendation(risk)
    print(f"  Sample {i+1} | Predicted: {preds_sample[i]} ({class_names[preds_sample[i]]}) | Risk Level: {risk}")
    print(f"    - Alert/Recommendation: {rec}")
print()

# Compute final confusion matrix (of the global aggregated model)
# Re-set client weights to global weights first to ensure confusion matrix represents the global model
for m in clients:
    m.set_weights(global_coef, global_intercept)

all_y_true = []
all_y_pred = []
for m in clients:
    X_te, y_te = m.get_test_data()
    all_y_pred.extend(m.model.predict(X_te))
    all_y_true.extend(y_te)

cm = confusion_matrix(all_y_true, all_y_pred, labels=[0, 1, 2, 3])

print("Combined Confusion Matrix (Global Aggregated Model):")
print(f"  {'Actual / Predicted':<20} {'Normal':<10} {'Mild':<10} {'Moderate':<10} {'Severe':<10}")
print(f"  {'-'*64}")
for i, name in enumerate(class_names):
    print(f"  {name:<20} {cm[i, 0]:<10} {cm[i, 1]:<10} {cm[i, 2]:<10} {cm[i, 3]:<10}")
print()

# ── STEP 6: Generate and Save Visualizations ──
print("Generating final training and accuracy plots...")
BG_COLOR = 'white'
TEXT_COLOR = '#202124'
BORDER_GRAY = '#dadce0'

# Plot 1: Federated learning progress over rounds
fig, ax = plt.subplots(figsize=(6.8, 6.0), facecolor=BG_COLOR)
ax.set_facecolor(BG_COLOR)

history_acc_pct = [acc * 100 for acc in history_acc]
mean_local_acc = np.mean(local_accuracies) * 100
rounds = list(range(1, ROUNDS + 1))

# Shaded area under the line
ax.fill_between(rounds, history_acc_pct, [mean_local_acc] * ROUNDS, alpha=0.06, color='#1a73e8')

# Baseline line
ax.axhline(mean_local_acc, color='#d93025', ls='--', lw=2.0,
           label=f'Mean Local Baseline (Standalone Avg) = {mean_local_acc:.2f}%')

ax.annotate(f'Local Baseline\n{mean_local_acc:.2f}%',
            xy=(ROUNDS + 0.05, mean_local_acc), fontsize=9, color='#d93025',
            va='center', ha='left', fontweight='bold')

# Main accuracy line
ax.plot(rounds, history_acc_pct,
        marker='o', color='#1a73e8', lw=2.8, markersize=10,
        markerfacecolor='white', markeredgecolor='#1a73e8', markeredgewidth=2.5,
        label='FedAvg Global Model Accuracy (Federated Learning)',
        zorder=5)

# Annotate each data point dynamically
for r, acc in zip(rounds, history_acc_pct):
    off = 0.12 if acc < mean_local_acc else -0.16
    ax.annotate(f'{acc:.2f}%',
                xy=(r, acc),
                xytext=(r, acc + off),
                textcoords='data',
                fontsize=9.5, color=TEXT_COLOR, fontweight='bold',
                ha='center',
                bbox=dict(boxstyle='round,pad=0.25', facecolor='white',
                          edgecolor=BORDER_GRAY, alpha=0.9))

# Gap annotation at final round
final_acc = history_acc_pct[-1]
ax.annotate('', xy=(ROUNDS + 0.25, mean_local_acc), xytext=(ROUNDS + 0.25, final_acc),
            arrowprops=dict(arrowstyle='<->', color='#5f6368', lw=1.2))
ax.text(ROUNDS + 0.32, (mean_local_acc + final_acc) / 2,
        f'Gap\n{mean_local_acc - final_acc:.2f}%',
        fontsize=8, color='#5f6368', va='center')

# Title and Labels
if use_dp:
    ax.set_title(f'Federated Learning — Accuracy Progress (Total Epsilon = {global_epsilon})', 
                 fontsize=12, fontweight='bold', color=TEXT_COLOR, pad=18)
else:
    ax.set_title('Federated Learning — Accuracy Progress (DP Disabled)', 
                 fontsize=12, fontweight='bold', color=TEXT_COLOR, pad=18)

ax.set_xlabel('Communication Round', fontsize=10, color=TEXT_COLOR, labelpad=10)
ax.set_ylabel('Combined Test Accuracy (%)', fontsize=10, color=TEXT_COLOR, labelpad=10)

ax.set_xticks(rounds)
ax.set_xticklabels([f'Round {r}' for r in rounds], fontsize=9.5, color=TEXT_COLOR)
ax.tick_params(axis='y', labelsize=9.5, labelcolor=TEXT_COLOR)
ax.spines[['top', 'right']].set_visible(False)
ax.spines[['left', 'bottom']].set_color(BORDER_GRAY)

# Grid and limits
ax.grid(color=BORDER_GRAY, ls='--', lw=0.8, alpha=0.7)
ax.set_xlim(0.5, ROUNDS + 0.65)
ax.set_ylim(min(history_acc_pct) - 1.5, max(max(history_acc_pct), mean_local_acc) + 1.5)

# Legend
ax.legend(loc='lower right', facecolor='white', edgecolor=BORDER_GRAY,
          labelcolor=TEXT_COLOR, fontsize=9, frameon=True)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'plot_accuracy.png'), dpi=300, facecolor=BG_COLOR, bbox_inches='tight')
plt.close()


# Plot 2: Laplace Noise PDF visualisation
fig, ax = plt.subplots(figsize=(7.0, 5.0), facecolor=BG_COLOR)
ax.set_facecolor(BG_COLOR)

if use_dp:
    from scipy.stats import laplace as laplace_dist
    if args.dp_mode == "output":
        scales = []
        for m in clients:
            R_CLIP = 5.0
            R = min(np.max(np.linalg.norm(m.X_train, axis=1)), R_CLIP)
            sensitivity = (2.0 * R) / (m.get_train_size() * m.alpha)
            # Use epsilon_per_round to plot the actual per-round noise scale
            scale = sensitivity / epsilon_per_round
            scales.append(scale)
        max_scale = max(scales)
        x_limit = max(3.0 * max_scale, 0.5)
        x_range = np.linspace(-x_limit, x_limit, 1000)

        colors = ['#1a73e8', '#e8710a', '#1e8e3e'] # Blue, Orange, Green
        line_styles = ['-', '--', ':']             # Solid, Dashed, Dotted
        line_widths = [3.5, 2.3, 1.6]              # Nested widths so all layers are visible

        for idx, scale in enumerate(scales):
            pdf = laplace_dist.pdf(x_range, loc=0, scale=scale)
            ax.plot(x_range, pdf, 
                    color=colors[idx], 
                    linestyle=line_styles[idx], 
                    linewidth=line_widths[idx], 
                    label=f"{names[idx]} (Scale b = {scale:.3f})")
            ax.fill_between(x_range, pdf, alpha=0.04, color=colors[idx])

        max_peak = max([laplace_dist.pdf(0, loc=0, scale=s) for s in scales])
        ax.set_title("PDF of Laplace DP Noise (Model Weights)", fontsize=12, fontweight='bold', color=TEXT_COLOR, pad=18)
        ax.set_xlabel("Noise Value Added to Model Weights", fontsize=10, color=TEXT_COLOR, labelpad=10)
        ax.set_xlim(-x_limit, x_limit)
        ax.set_ylim(-0.02 * max_peak, max_peak * 1.18)
    else:
        # Input perturbation: plot noise added to features based on sensitivity
        # Standardized range is [-3, 3], Delta = 6.0
        # Medium sensitivity: multiplier = 1.0, scale = Delta / epsilon
        # High sensitivity: multiplier = 1.5, scale = (Delta * 1.5) / epsilon
        # Low sensitivity: multiplier = 0.1, scale = (Delta * 0.1) / epsilon
        x_range = np.linspace(-30.0, 30.0, 1000)
        
        scale_med = 6.0 / global_epsilon
        scale_high = (6.0 * 1.5) / global_epsilon
        scale_low = (6.0 * 0.1) / global_epsilon
        
        pdf_med = laplace_dist.pdf(x_range, loc=0, scale=scale_med)
        pdf_high = laplace_dist.pdf(x_range, loc=0, scale=scale_high)
        pdf_low = laplace_dist.pdf(x_range, loc=0, scale=scale_low)
        
        ax.plot(x_range, pdf_low, color='#1e8e3e', linestyle=':', linewidth=2.0, label=f"Low Sensitivity (Scale b = {scale_low:.3f})")
        ax.fill_between(x_range, pdf_low, alpha=0.04, color='#1e8e3e')
        
        ax.plot(x_range, pdf_med, color='#1a73e8', linestyle='-', linewidth=2.5, label=f"Medium Sensitivity (Scale b = {scale_med:.3f})")
        ax.fill_between(x_range, pdf_med, alpha=0.04, color='#1a73e8')
        
        ax.plot(x_range, pdf_high, color='#e8710a', linestyle='--', linewidth=2.0, label=f"High Sensitivity (Scale b = {scale_high:.3f})")
        ax.fill_between(x_range, pdf_high, alpha=0.04, color='#e8710a')
        
        ax.set_title("PDF of Laplace DP Noise (Input Features)", fontsize=12, fontweight='bold', color=TEXT_COLOR, pad=18)
        ax.set_xlabel("Noise Value Added to Standardized Features", fontsize=10, color=TEXT_COLOR, labelpad=10)
        ax.set_xlim(-25.0, 25.0)
        ax.set_ylim(-0.01, max(pdf_low[500], pdf_med[500], pdf_high[500]) + 0.1)

    ax.set_ylabel("Probability Density", fontsize=10, color=TEXT_COLOR, labelpad=10)
    ax.grid(color=BORDER_GRAY, ls='--', lw=0.8, alpha=0.7)
    ax.tick_params(axis='both', labelsize=9.5, labelcolor=TEXT_COLOR)
    ax.spines[['top', 'right']].set_visible(False)
    ax.spines[['left', 'bottom']].set_color(BORDER_GRAY)
    ax.legend(loc='upper right', facecolor='white', edgecolor=BORDER_GRAY,
              labelcolor=TEXT_COLOR, fontsize=9.5, frameon=True)
else:
    ax.text(0.5, 0.5, "Differential Privacy is Disabled\n(No Noise Added to Features or Weights)", 
            color=TEXT_COLOR, ha='center', va='center', fontsize=11, fontweight='bold', style='italic')
    ax.set_title("Laplace DP Noise PDF (DP Disabled)", color=TEXT_COLOR, fontsize=12, fontweight='bold')
    ax.set_axis_off()

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'plot_dp_laplace.png'), dpi=300, facecolor=BG_COLOR, bbox_inches='tight')
plt.close()


# Plot 3: Dataset split stats (stacked bar chart showing Train vs Test sizes)
fig, ax = plt.subplots(figsize=(6.0, 5.2), facecolor=BG_COLOR)
ax.set_facecolor(BG_COLOR)

client_names = ["Machine 1", "Machine 2", "Machine 3"]
train_sizes = [m.get_train_size() for m in clients]
test_sizes = [len(m.get_test_data()[1]) for m in clients]

width = 0.42
bars_train = ax.bar(client_names, train_sizes, color='#1a73e8', width=width, 
                    label='Training Data (80%)', edgecolor='#1357b3', lw=1.2)
bars_test = ax.bar(client_names, test_sizes, bottom=train_sizes, color='#34a853', width=width, 
                   label='Test Data (20%)', edgecolor='#247d3c', lw=1.2)

for bt, val_t in zip(bars_train, train_sizes):
    ax.text(bt.get_x() + bt.get_width()/2.0, 
            val_t / 2.0, 
            f"{val_t}\n(80%)", 
            ha='center', va='center', 
            color='white', fontweight='bold', fontsize=9.0)

for bt_train, bt_test, val_te in zip(bars_train, bars_test, test_sizes):
    ax.text(bt_test.get_x() + bt_test.get_width()/2.0, 
            bt_train.get_height() + (val_te / 2.0), 
            f"{val_te}\n(20%)", 
            ha='center', va='center', 
            color='white', fontweight='bold', fontsize=9.0)

ax.set_title("Dataset Records Distribution Across Machines", fontsize=11.5, fontweight='bold', color=TEXT_COLOR, pad=18)
ax.set_ylabel("Number of Records", fontsize=10, fontweight='bold', color=TEXT_COLOR, labelpad=10)
ax.spines[['top', 'right']].set_visible(False)
ax.spines[['left', 'bottom']].set_color(BORDER_GRAY)
ax.tick_params(axis='x', colors=TEXT_COLOR, labelsize=9.5)
ax.tick_params(axis='y', colors=TEXT_COLOR, labelsize=9)
ax.grid(axis='y', color=BORDER_GRAY, ls='--', lw=0.8, alpha=0.7)
ax.legend(loc='upper right', facecolor='white', edgecolor=BORDER_GRAY, 
          labelcolor=TEXT_COLOR, fontsize=9.0, frameon=True)
ax.set_ylim(0, max(train_sizes) + max(test_sizes) + 300)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'plot_dataset_sizes.png'), dpi=300, facecolor=BG_COLOR, bbox_inches='tight')
plt.close()


# Plot 4: Confusion Matrix Plot
fig, ax = plt.subplots(figsize=(6.2, 5.8), facecolor=BG_COLOR)
ax.set_facecolor(BG_COLOR)
plt.subplots_adjust(left=0.18, right=0.96, top=0.88, bottom=0.15)

wrapped_class_names = ['Normal', 'Mild\nEvent', 'Moderate\nEvent', 'Severe\nEvent']
total_per_class = cm.sum(axis=1)

nrows, ncols = cm.shape
for i in range(nrows):
    for j in range(ncols):
        val = cm[i, j]
        pct = (val / total_per_class[i]) * 100 if total_per_class[i] > 0 else 0
        
        if i == j:
            facecolor = plt.cm.Blues(0.15 + 0.7 * (pct / 100.0))
            text_color = 'white' if pct > 50 else TEXT_COLOR
        else:
            if val > 0:
                facecolor = '#fce8e6'
                text_color = '#c5221f'
            else:
                facecolor = 'white'
                text_color = '#9aa0a6'
        
        rect = plt.Rectangle((j - 0.5, i - 0.5), 1.0, 1.0, 
                             facecolor=facecolor, edgecolor=BORDER_GRAY, lw=1)
        ax.add_patch(rect)
        
        if val > 0:
            if i == j:
                label_text = f"{val}\n({pct:.1f}%)"
            else:
                label_text = f"{val}"
            font_weight = 'bold'
        else:
            label_text = "0"
            font_weight = 'normal'
            
        ax.text(j, i, label_text, ha='center', va='center', 
                color=text_color, fontsize=10, fontweight=font_weight)

ax.set_xlim(-0.5, ncols - 0.5)
ax.set_ylim(nrows - 0.5, -0.5)
ax.set_xticks(range(ncols))
ax.set_yticks(range(nrows))
ax.set_xticklabels(wrapped_class_names, fontsize=9.5, color=TEXT_COLOR)
ax.set_yticklabels(wrapped_class_names, fontsize=9.5, color=TEXT_COLOR, va='center')
ax.set_xlabel('Predicted Class (Model Output)', fontsize=10.5, fontweight='bold', color=TEXT_COLOR, labelpad=8)
ax.set_ylabel('Actual Class (Ground Truth)', fontsize=10.5, fontweight='bold', color=TEXT_COLOR, labelpad=8)

for spine in ax.spines.values():
    spine.set_visible(False)

ax.hlines(np.arange(nrows+1)-0.5, -0.5, ncols-0.5, colors=BORDER_GRAY, linewidths=1)
ax.vlines(np.arange(ncols+1)-0.5, -0.5, nrows-0.5, colors=BORDER_GRAY, linewidths=1)
plt.title('Confusion Matrix — Federated Global Model', fontsize=12, fontweight='bold', color=TEXT_COLOR, pad=15)

plt.tight_layout()
fig.savefig(os.path.join(OUT_DIR, 'plot_confusion_matrix.png'), dpi=300, facecolor=BG_COLOR, bbox_inches='tight', pad_inches=0.05)
plt.close()

print("Plots saved: plot_accuracy.png, plot_dp_laplace.png, plot_dataset_sizes.png, plot_confusion_matrix.png")
print("Done.")
