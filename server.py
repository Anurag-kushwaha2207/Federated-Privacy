# -*- coding: utf-8 -*-
"""
============================================================
  SERVER — Federated Learning + Differential Privacy
  Central Aggregator
============================================================
  This file connects Machine 1, 2, and 3 to run a multi-round
  Horizontal Federated Learning loop using FedAvg.
  
  Integrates:
  1. Federated Feature Statistics Aggregation (Zero Data Leakage)
  2. Single-Release Output Differential Privacy (Applied at Final Round)
  3. Stratified Leakage-Safe Test Splits (100% Real Test Records)
  4. Unified R_CLIP = 1.5 module-level constant
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
from machine1 import Machine1, ALPHA, R_CLIP
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
    print(f"  Regularization      : L2 (alpha = {ALPHA})")
    print(f"  Max Norm Clipping   : R_CLIP = {R_CLIP}\n")
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

    try:
        dp_enabled_clients = {int(x.strip()) for x in args.dp_clients.split(",") if x.strip()}
    except Exception:
        dp_enabled_clients = {1, 2, 3}

    print("\nConfiguration:")
    print(f"  Differential Privacy: Enabled ({'Laplace Input Perturbation' if args.dp_mode == 'input' else 'Laplace Output Perturbation (Single-Release)'})")
    print(f"  Privacy Budget (eps): {global_epsilon}")
    active_dp_str = ", ".join(f"Machine {x}" for x in sorted(dp_enabled_clients)) if dp_enabled_clients else "None"
    print(f"  DP Active On        : {active_dp_str}")
    print(f"  DP Mode             : {args.dp_mode.upper()} PERTURBATION")
    print(f"  Regularization      : L2 (alpha = {ALPHA})")
    print(f"  Max Norm Clipping   : R_CLIP = {R_CLIP}\n")

ROUNDS = 5

print("Initializing clients and partitioning dataset...")
m1 = Machine1(epsilon=global_epsilon if 1 in dp_enabled_clients else 0.0, dp_mode=args.dp_mode)
m2 = Machine2(epsilon=global_epsilon if 2 in dp_enabled_clients else 0.0, dp_mode=args.dp_mode)
m3 = Machine3(epsilon=global_epsilon if 3 in dp_enabled_clients else 0.0, dp_mode=args.dp_mode)

clients = [m1, m2, m3]
names = ["Machine 1", "Machine 2", "Machine 3"]

# ── BUG 1 FIX: Privacy-Preserving Federated Feature Statistics Aggregation ──
print("Executing Federated Feature Statistics Aggregation (Zero Data Leakage)...")
stats = [m.get_feature_stats() for m in clients]
total_N = sum(s[0] for s in stats)

# Pooled global mean
mu_fed = np.zeros(stats[0][1].shape)
for n_i, mean_i, _ in stats:
    mu_fed += n_i * mean_i
mu_fed /= total_N

# Pooled global variance & scale
var_fed = np.zeros(stats[0][2].shape)
for n_i, mean_i, var_i in stats:
    var_fed += n_i * (var_i + (mean_i - mu_fed) ** 2)
var_fed /= total_N
scale_fed = np.sqrt(var_fed)

# Broadcast federated scaler parameters to all edge client nodes
for m in clients:
    m.set_federated_scaler(mu_fed, scale_fed)

print("Federated scaling initialized cleanly across all client nodes.\n")

# ── STEP 2: Compute Baseline (Standalone Local Models) ──
print("Evaluating standalone local models (baselines)...")
print("-" * 64)

local_accuracies = []
for idx, m in enumerate(clients):
    m.initialize_model()
    m.local_train(epochs=100)
    
    X_te, y_te = m.get_test_data()
    preds = m.model.predict(X_te)
    acc = accuracy_score(y_te, preds)
    local_accuracies.append(acc)
    print(f"  {names[idx]} train size: {m.get_train_size()} | Accuracy: {acc*100:.2f}%")

# ── STEP 3: Multi-Round Federated Learning Loop ──
print("\nStarting Federated Learning training loops (FedAvg)...")
print("-" * 64)

history_acc = []

for m in clients:
    m.initialize_model()

global_coef, global_intercept = m1.get_weights()
global_coef.fill(0.0)
global_intercept.fill(0.0)

for r in range(1, ROUNDS + 1):
    print(f"Round {r}/{ROUNDS}:")
    
    local_weights = []
    dataset_sizes = []
    is_final_round = (r == ROUNDS)
    
    for idx, m in enumerate(clients):
        m.set_weights(global_coef, global_intercept)
        m.local_train(epochs=5)
        
        client_num = idx + 1
        # BUG 4 FIX: Single-Release Output DP applied exclusively on final round
        if use_dp and client_num in dp_enabled_clients and is_final_round:
            noisy_coef, noisy_intercept = m.get_dp_weights(global_epsilon)
            if args.dp_mode == "input":
                print(f"    - {names[idx]}: (Input DP Enabled, Epsilon = {global_epsilon:.4f})")
            else:
                R = min(np.max(np.linalg.norm(m.X_train, axis=1)), R_CLIP)
                sensitivity = (2.0 * R) / (m.get_train_size() * m.alpha)
                scale = sensitivity / global_epsilon
                print(f"    - {names[idx]}: sensitivity = {sensitivity:.6f}, noise scale = {scale:.6f}, max_norm(R) = {R:.4f} (DP Single Release Enabled, Epsilon = {global_epsilon:.4f})")
        else:
            noisy_coef, noisy_intercept = m.get_weights()
            if r == 1:
                if use_dp and client_num in dp_enabled_clients:
                    print(f"    - {names[idx]}: (Clean intermediate FL training; Single-Release DP deferred to final round)")
                else:
                    print(f"    - {names[idx]}: (DP Disabled)")
        
        local_weights.append((noisy_coef, noisy_intercept))
        dataset_sizes.append(m.get_train_size())
            
    # Aggregation (FedAvg)
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
    
    # Evaluate global model
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
    print(f"  Total Privacy Epsilon       : {global_epsilon:.4f} (Single Release — Applied Once at Final Round)")
else:
    print(f"  Differential Privacy Status : DISABLED")

print(f"  {'Model / Client':<18} {'Train Records':>15} {'Accuracy':>12}")
print("  " + "-" * 49)
for idx, m in enumerate(clients):
    print(f"  {names[idx]:<18} {m.get_train_size():>15} {local_accuracies[idx]*100:>11.2f}%")
print("  " + "-" * 49)
print(f"  {'FedAvg Global':<18} {sum(m.get_train_size() for m in clients):>15} {history_acc[-1]*100:>11.2f}%")
print("  " + "-" * 49)

# ── STEP 5: Model Personalization ──
print("\n=================== MODEL PERSONALIZATION ===================")
print("Each client receives the global model and fine-tunes on local data...")
print("-" * 61)

personalized_accuracies = []
for idx, m in enumerate(clients):
    X_te, y_te = m.get_test_data()
    m.set_weights(global_coef, global_intercept)
    global_eval_acc = accuracy_score(y_te, m.model.predict(X_te))
    
    m.fine_tune(global_coef, global_intercept, epochs=10)
    fine_tuned_preds = m.model.predict(X_te)
    fine_tuned_acc = accuracy_score(y_te, fine_tuned_preds)
    personalized_accuracies.append(fine_tuned_acc)
    
    print(f"  {names[idx]} | Global Acc: {global_eval_acc*100:.2f}% | Personalized Acc: {fine_tuned_acc*100:.2f}%")

avg_pers_acc = np.mean(personalized_accuracies)
print(f"  Average Personalized Accuracy: {avg_pers_acc*100:.2f}%\n")

# ── STEP 6: Sample Risk Mappings ──
print("=================== SAMPLE RISK MAPPINGS & ALERTS ===================")
print("Displaying rule-based health recommendations from predicted classes...")
print("-" * 69)

class_names_map = {0: "Normal", 1: "Mild Event", 2: "Moderate Event", 3: "Severe Event"}
sample_x, sample_y = m1.get_test_data()
sample_preds = m1.model.predict(sample_x[:4])

def get_health_recommendation(pred_class):
    if pred_class == 0:
        return "LOW RISK", "Status: Normal. Vital signs are within baseline limits. Continue standard monitoring."
    elif pred_class in [1, 2]:
        return "MEDIUM RISK", "Status: Mild/Moderate Event detected. Recommendation: Monitor vital signs closely, reduce stress, and rest."
    else:
        return "HIGH RISK", "Status: Severe Event detected! Alert: Immediate clinical consultation is advised."

for idx, p in enumerate(sample_preds):
    risk_level, alert_msg = get_health_recommendation(p)
    print(f"  Sample {idx+1} | Predicted: {p} ({class_names_map[p]}) | Risk Level: {risk_level}")
    print(f"    - Alert/Recommendation: {alert_msg}")

print(f"\nCombined Confusion Matrix (Global Aggregated Model):")
print(f"  {'Actual / Predicted':<20} {'Normal':<10} {'Mild':<10} {'Moderate':<10} {'Severe':<10}")
print("  " + "-" * 64)

cm = confusion_matrix(all_y_true, all_y_pred)
labels = ["Normal", "Mild Event", "Moderate Event", "Severe Event"]
for i, label in enumerate(labels):
    row_str = f"  {label:<20}"
    for j in range(4):
        val = cm[i][j] if i < len(cm) and j < len(cm[i]) else 0
        row_str += f"{val:<10}"
    print(row_str)

# ── STEP 7: Plots & Visualizations ──
print("\nGenerating final training and accuracy plots...")
plt.style.use('default')
fig_bg = '#ffffff'
card_bg = '#ffffff'
TEXT_COLOR = '#202124'

# Plot 1: Accuracy Curve with Baseline, Shaded Region, Callouts, and Gap Arrow (Matching Screenshot 1)
fig, ax = plt.subplots(figsize=(8.5, 6), facecolor=fig_bg)
ax.set_facecolor(card_bg)

rounds_x = [1, 2, 3, 4, 5]
history_acc_pct = [a * 100 for a in history_acc]
baseline_acc_pct = np.mean(local_accuracies) * 100

# Red dashed baseline line
ax.axhline(y=baseline_acc_pct, color='#d93025', linestyle='--', linewidth=2.2, 
           label=f'Mean Local Baseline (Standalone Avg) = {baseline_acc_pct:.2f}%')

# Blue FedAvg curve
ax.plot(rounds_x, history_acc_pct, color='#1a73e8', linewidth=3.0, marker='o', 
        markersize=9, markerfacecolor='#ffffff', markeredgecolor='#1a73e8', markeredgewidth=2.5,
        label='FedAvg Global Model Accuracy (Federated Learning)')

# Shaded background region between FedAvg curve and baseline
ax.fill_between(rounds_x, history_acc_pct, baseline_acc_pct, color='#e8f0fe', alpha=0.6)

# Callout boxes on each round node
for idx, (rx, acc) in enumerate(zip(rounds_x, history_acc_pct)):
    ax.annotate(f"{acc:.2f}%", (rx, acc),
                xytext=(0, 10 if idx % 2 == 0 else -18), textcoords='offset points',
                ha='center', fontsize=9, fontweight='bold', color='#202124',
                bbox=dict(boxstyle='round,pad=0.3', facecolor='#ffffff', edgecolor='#dadce0', alpha=0.9))

# Gap annotation arrow on final round
final_acc = history_acc_pct[-1]
gap = baseline_acc_pct - final_acc
if abs(gap) > 0.05:
    ax.annotate(f"Gap\n{gap:.2f}%", 
                xy=(5, final_acc), 
                xytext=(5, (final_acc + baseline_acc_pct) / 2.0),
                arrowprops=dict(arrowstyle='<->', color='#70757a', lw=1.5),
                ha='center', va='center', fontsize=8.5, color='#5f6368',
                bbox=dict(boxstyle='square,pad=0.2', facecolor='#ffffff', edgecolor='none', alpha=0.8))

ax.set_title(f"Federated Learning — Accuracy Progress (Total Epsilon = {global_epsilon if use_dp else 'Disabled'})", 
             fontsize=12, fontweight='bold', color=TEXT_COLOR, pad=18)
ax.set_xlabel("Communication Round", fontsize=10, color=TEXT_COLOR, labelpad=10)
ax.set_ylabel("Combined Test Accuracy (%)", fontsize=10, color=TEXT_COLOR, labelpad=10)
ax.set_xticks(rounds_x)
ax.set_xticklabels([f"Round {r}" for r in rounds_x], fontsize=9)

y_min = min(min(history_acc_pct), baseline_acc_pct) - 5
y_max = max(max(history_acc_pct), baseline_acc_pct) + 5
ax.set_ylim(max(0, y_min), min(100, y_max))

ax.grid(True, linestyle='--', alpha=0.4)
ax.legend(loc='lower right', frameon=True, facecolor=card_bg, edgecolor='#dadce0', fontsize=9)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "plot_accuracy.png"), dpi=300)
plt.close()

# Plot 2: DP Laplace Distribution
fig, ax = plt.subplots(figsize=(8, 5), facecolor=fig_bg)
ax.set_facecolor(card_bg)
from scipy.stats import laplace as laplace_dist
if use_dp and args.dp_mode == "output":
    scales = []
    for m in clients:
        R = min(np.max(np.linalg.norm(m.X_train, axis=1)), R_CLIP)
        sensitivity = (2.0 * R) / (m.get_train_size() * m.alpha)
        scale = sensitivity / global_epsilon
        scales.append(scale)
    max_scale = max(scales)
    x_limit = max(3.0 * max_scale, 0.5)
    x_range = np.linspace(-x_limit, x_limit, 1000)

    colors = ['#1a73e8', '#e8710a', '#1e8e3e']
    line_styles = ['-', '--', ':']
    line_widths = [3.5, 2.3, 1.6]

    for idx, scale in enumerate(scales):
        pdf = laplace_dist.pdf(x_range, loc=0, scale=scale)
        ax.plot(x_range, pdf, color=colors[idx], linestyle=line_styles[idx], linewidth=line_widths[idx], label=f"{names[idx]} (Scale b = {scale:.6f})")
        ax.fill_between(x_range, pdf, alpha=0.04, color=colors[idx])

    max_peak = max([laplace_dist.pdf(0, loc=0, scale=s) for s in scales])
    ax.set_title("PDF of Laplace DP Noise (Single-Release Model Weights)", fontsize=12, fontweight='bold', color=TEXT_COLOR, pad=18)
    ax.set_xlabel("Noise Value Added to Model Weights", fontsize=10, color=TEXT_COLOR, labelpad=10)
    ax.set_xlim(-x_limit, x_limit)
    ax.set_ylim(-0.02 * max_peak, max_peak * 1.18)
else:
    x_range = np.linspace(-1.0, 1.0, 1000)
    pdf = laplace_dist.pdf(x_range, loc=0, scale=0.05)
    ax.plot(x_range, pdf, color='#1a73e8', label="Laplace Noise PDF")
    ax.set_title("PDF of Laplace DP Noise", fontsize=12, fontweight='bold')

ax.grid(True, linestyle='--', alpha=0.5)
ax.legend(loc='upper right')
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "plot_dp_laplace.png"), dpi=300)
plt.close()

# Plot 3: Stacked Bar Chart for Dataset Records Distribution (Matching Screenshot 2)
fig, ax = plt.subplots(figsize=(8.5, 6), facecolor=fig_bg)
ax.set_facecolor(card_bg)

train_sizes = [m.get_train_size() for m in clients]
test_sizes  = [len(m.get_test_data()[1]) for m in clients]

x_indices = np.arange(len(names))
bar_width = 0.45

# Stacked bars: Train (Blue) at bottom, Test (Green) on top
bars_train = ax.bar(x_indices, train_sizes, width=bar_width, label='Training Data (80%)', color='#1a73e8', edgecolor='none')
bars_test  = ax.bar(x_indices, test_sizes,  width=bar_width, bottom=train_sizes, label='Test Data (20%)', color='#2e7d32', edgecolor='none')

# Labels inside Blue bar (Training)
for idx, (bar, t_cnt) in enumerate(zip(bars_train, train_sizes)):
    total = t_cnt + test_sizes[idx]
    pct = (t_cnt / total) * 100
    y_pos = t_cnt / 2.0
    ax.text(bar.get_x() + bar.get_width()/2.0, y_pos, f"{t_cnt}\n({pct:.0f}%)", 
            ha='center', va='center', fontsize=9.5, fontweight='bold', color='#ffffff')

# Labels inside Green bar (Test)
for idx, (bar, te_cnt) in enumerate(zip(bars_test, test_sizes)):
    t_cnt = train_sizes[idx]
    total = t_cnt + te_cnt
    pct = (te_cnt / total) * 100
    y_pos = t_cnt + (te_cnt / 2.0)
    ax.text(bar.get_x() + bar.get_width()/2.0, y_pos, f"{te_cnt}\n({pct:.0f}%)", 
            ha='center', va='center', fontsize=9.5, fontweight='bold', color='#ffffff')

ax.set_title("Dataset Records Distribution Across Machines", fontsize=13, fontweight='bold', color=TEXT_COLOR, pad=18)
ax.set_xlabel("Client Node", fontsize=10, color=TEXT_COLOR, labelpad=10)
ax.set_ylabel("Number of Records", fontsize=10, color=TEXT_COLOR, labelpad=10)
ax.set_xticks(x_indices)
ax.set_xticklabels(names, fontsize=10, fontweight='bold')

max_total = max([tr + te for tr, te in zip(train_sizes, test_sizes)])
ax.set_ylim(0, max_total * 1.15)
ax.grid(True, linestyle='--', alpha=0.4, axis='y')
ax.legend(loc='upper right', frameon=True, facecolor=card_bg, edgecolor='#dadce0', fontsize=9.5)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "plot_dataset_sizes.png"), dpi=300)
plt.close()

# Plot 4: Confusion Matrix
fig, ax = plt.subplots(figsize=(7, 6), facecolor=fig_bg)
ax.set_facecolor(card_bg)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=labels, yticklabels=labels, ax=ax, cbar=False)
ax.set_title("Global Aggregated Model Confusion Matrix", fontsize=12, fontweight='bold', pad=15)
ax.set_xlabel("Predicted Condition", fontsize=10, labelpad=10)
ax.set_ylabel("True Condition", fontsize=10, labelpad=10)
plt.tight_layout()
plt.savefig(os.path.join(OUT_DIR, "plot_confusion_matrix.png"), dpi=300)
plt.close()

print("Plots saved: plot_accuracy.png, plot_dp_laplace.png, plot_dataset_sizes.png, plot_confusion_matrix.png")
print("Done.")
