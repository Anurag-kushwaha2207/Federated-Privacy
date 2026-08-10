# -*- coding: utf-8 -*-
"""
Standalone Client Execution Script

Evaluates Machine 1, Machine 2, and Machine 3 independently without federated aggregation.
"""

import sys
import os
import argparse
import warnings
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

np.random.seed(42)

sys.stdout.reconfigure(encoding='utf-8', errors='replace')
warnings.filterwarnings('ignore')

from machine1 import Machine1
from machine2 import Machine2
from machine3 import Machine3

def main():
    parser = argparse.ArgumentParser(description="Standalone Client Evaluation")
    parser.add_argument("--no-dp", action="store_true", help="Run without Differential Privacy")
    parser.add_argument("-e", "--epsilon", type=float, default=None, help="Privacy Budget (Epsilon)")
    parser.add_argument("--dp-mode", type=str, choices=["input", "output"], default="output", help="DP Perturbation stage: 'input' or 'output'")
    args, unknown = parser.parse_known_args()

    use_dp = not args.no_dp

    print("================================================================")
    print("Standalone Client Execution")
    if not use_dp:
        global_epsilon = 1.0
        print("Differential Privacy: Disabled")
    else:
        if args.epsilon is not None:
            global_epsilon = args.epsilon
        else:
            try:
                user_eps = input("Enter Privacy Budget (Epsilon) [default: 1.0]: ").strip()
                global_epsilon = float(user_eps) if user_eps else 1.0
            except Exception:
                global_epsilon = 1.0

        if global_epsilon <= 0:
            print("Error: Epsilon must be strictly greater than 0.")
            sys.exit(1)

        print(f"Differential Privacy: Enabled (Epsilon = {global_epsilon}, Mode = {args.dp_mode.upper()})")
    print("================================================================\n")

    eps_val = global_epsilon if use_dp else 0.0
    m1 = Machine1(epsilon=eps_val, dp_mode=args.dp_mode)
    m2 = Machine2(epsilon=eps_val, dp_mode=args.dp_mode)
    m3 = Machine3(epsilon=eps_val, dp_mode=args.dp_mode)

    clients = [m1, m2, m3]
    names = ["Machine 1", "Machine 2", "Machine 3"]

    print("Evaluating standalone models...")
    print("-" * 55)

    local_accuracies = []
    all_y_true = []
    all_y_pred = []

    for idx, m in enumerate(clients):
        m.initialize_model()
        # Train for 100 epochs locally
        m.local_train(epochs=100)

        if use_dp:
            coef, intercept = m.get_dp_weights()
            m.set_weights(coef, intercept)
        else:
            coef, intercept = m.get_weights()

        X_te, y_te = m.get_test_data()
        preds = m.model.predict(X_te)
        acc = accuracy_score(y_te, preds)

        local_accuracies.append(acc)
        all_y_true.extend(y_te)
        all_y_pred.extend(preds)

        print(f"  {names[idx]:<12} | Train Samples: {m.get_train_size():<5} | Test Accuracy: {acc*100:.2f}%")

    print("\n=================== LOCAL MODELS SUMMARY ===================")
    print(f"  {'Client':<15} {'Train Samples':>15} {'Test Accuracy':>15}")
    print(f"  {'-'*47}")
    for idx in range(len(clients)):
        sample_cnt = clients[idx].get_train_size()
        print(f"  {names[idx]:<15} {sample_cnt:>15} {local_accuracies[idx]*100:>14.2f}%")
    print(f"  {'-'*47}\n")

    # Map predicted class to health risk level
    def map_to_risk_level(pred_class):
        if pred_class == 0:
            return "LOW RISK"
        elif pred_class in [1, 2]:
            return "MEDIUM RISK"
        else:
            return "HIGH RISK"

    def get_health_recommendation(risk_level):
        if risk_level == "LOW RISK":
            return "Status: Normal. Recommendation: Maintain your daily exercise routine and healthy sleep patterns."
        elif risk_level == "MEDIUM RISK":
            return "Status: Mild/Moderate Event detected. Recommendation: Monitor your vital signs closely, reduce physical stress, and rest."
        else:
            return "Status: Severe Event detected! Alert: Immediate clinical consultation is advised. Avoid strenuous activities."

    class_names = ['Normal', 'Mild Event', 'Moderate Event', 'Severe Event']
    
    print("=================== SAMPLE RISK MAPPINGS & ALERTS ===================")
    for idx, m in enumerate(clients):
        X_sample, y_sample = m.get_test_data()
        preds_sample = m.model.predict(X_sample[:1])
        risk = map_to_risk_level(preds_sample[0])
        rec = get_health_recommendation(risk)
        print(f"  {names[idx]} | Pred: {preds_sample[0]} ({class_names[preds_sample[0]]}) | Risk: {risk}")
        print(f"    - Recommendation: {rec}")
    print()

    cm = confusion_matrix(all_y_true, all_y_pred, labels=[0, 1, 2, 3])

    print("Combined Confusion Matrix:")
    print(f"  {'Actual / Predicted':<20} {'Normal':<10} {'Mild':<10} {'Moderate':<10} {'Severe':<10}")
    print(f"  {'-'*64}")
    for i, name in enumerate(class_names):
        print(f"  {name:<20} {cm[i, 0]:<10} {cm[i, 1]:<10} {cm[i, 2]:<10} {cm[i, 3]:<10}")
    print()

if __name__ == "__main__":
    main()
