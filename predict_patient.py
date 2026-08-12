# -*- coding: utf-8 -*-
"""
Single-Patient Prediction & XAI Explanation Tool

Loads the pre-trained Federated Learning + Differential Privacy global model
from `trained_model.pkl` and provides instant condition prediction and SHAP 
local risk explanations for individual patient vitals.
"""

import os
import sys
import argparse
import joblib
import numpy as np
from sklearn.linear_model import SGDClassifier

# Import local explanation module
from explain import explain_prediction

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "trained_model.pkl")

def get_health_recommendation(pred_class):
    if pred_class == 0:
        return "LOW RISK", "Status: Normal. Vital signs are within baseline limits. Continue standard monitoring."
    elif pred_class in [1, 2]:
        return "MEDIUM RISK", "Status: Mild/Moderate Event detected. Recommendation: Monitor vital signs closely, reduce stress, and rest."
    else:
        return "HIGH RISK", "Status: Severe Event detected! Alert: Immediate clinical consultation is advised."

def main():
    parser = argparse.ArgumentParser(description="Single-Patient Health Prediction & XAI Explanation CLI")
    
    # 13 Raw Patient Input Features
    parser.add_argument("--heart_rate", type=float, default=75.0, help="Heart rate in bpm")
    parser.add_argument("--blood_oxygen", type=float, default=98.0, help="Blood oxygen SpO2 percentage")
    parser.add_argument("--blood_pressure_systolic", type=float, default=120.0, help="Systolic BP mmHg")
    parser.add_argument("--blood_pressure_diastolic", type=float, default=80.0, help="Diastolic BP mmHg")
    parser.add_argument("--glucose_level", type=float, default=95.0, help="Glucose level mg/dL")
    parser.add_argument("--body_temperature", type=float, default=98.6, help="Body temperature in Fahrenheit (°F)")
    parser.add_argument("--respiratory_rate", type=float, default=16.0, help="Respiratory rate breaths/min")
    parser.add_argument("--activity_level", type=float, default=0.5, help="Activity level score (0.0 to 1.0)")
    parser.add_argument("--sleep_quality", type=float, default=0.7, help="Sleep quality score (0.0 to 1.0)")
    parser.add_argument("--stress_level", type=float, default=0.3, help="Stress level score (0.0 to 1.0)")
    parser.add_argument("--hrv_sdnn", type=float, default=45.0, help="HRV SDNN ms")
    parser.add_argument("--steps_count", type=float, default=6000.0, help="Daily steps count")
    parser.add_argument("--calories_burned", type=float, default=2000.0, help="Daily calories burned")
    
    args = parser.parse_args()
    
    if not os.path.exists(MODEL_PATH):
        print(f"Error: Model file '{MODEL_PATH}' not found.")
        print("Please run 'python server.py' first to train and save the global model.")
        sys.exit(1)
        
    # Load trained model state & federated scaler parameters
    artifacts = joblib.load(MODEL_PATH)
    global_coef = artifacts["global_coef"]
    global_intercept = artifacts["global_intercept"]
    mu_fed = artifacts["mu_fed"]
    scale_fed = artifacts["scale_fed"]
    feature_names = artifacts["feature_names"]
    class_names = artifacts["class_names"]
    alpha = artifacts.get("alpha", 0.20)

    # 1. Feature Engineering: Compute non-linear interaction features
    hr = args.heart_rate
    spo2 = args.blood_oxygen
    bp_sys = args.blood_pressure_systolic
    bp_dia = args.blood_pressure_diastolic
    stress = args.stress_level

    hr_stress_ratio = hr * stress
    spo2_deficit = 100.0 - spo2
    bp_diff = bp_sys - bp_dia
    vital_risk_index = (hr / 70.0) + (spo2_deficit / 5.0) + (stress * 2.0)

    # Build 17-feature vector matching training order
    raw_features = np.array([[
        hr, spo2, bp_sys, bp_dia,
        args.glucose_level, args.body_temperature, args.respiratory_rate,
        args.activity_level, args.sleep_quality, stress,
        args.hrv_sdnn, args.steps_count, args.calories_burned,
        hr_stress_ratio, spo2_deficit, bp_diff, vital_risk_index
    ]], dtype=np.float64)

    # 2. Standardize features using stored federated scaling parameters
    X_scaled = (raw_features - mu_fed) / scale_fed

    # 3. Reconstruct trained linear SGD model
    model = SGDClassifier(loss='log_loss', penalty='l2', alpha=alpha, random_state=42)
    model.classes_ = np.array([0, 1, 2, 3])
    model.coef_ = global_coef.copy()
    model.intercept_ = global_intercept.copy()
    model.t_ = 1.0

    # 4. Generate SHAP background distribution (Standard normal in scaled space)
    rng = np.random.RandomState(42)
    X_background = rng.randn(100, 17)

    # 5. Predict and Explain
    pred_class, explanation_parts = explain_prediction(model, X_background, X_scaled[0], feature_names)
    risk_level, rec_msg = get_health_recommendation(pred_class)
    predicted_label = class_names[pred_class]

    # Clean display output
    print("\n==================================================")
    print("  PATIENT HEALTH PREDICTION & EXPLANATION REPORT  ")
    print("==================================================")
    print(f"Predicted Condition: {predicted_label}")
    print(f"Risk Level         : {risk_level}")
    print(f"Recommendation     : {rec_msg}")
    print(f"Primary Factors    : {', '.join(explanation_parts)}")
    print("==================================================\n")

if __name__ == "__main__":
    main()
