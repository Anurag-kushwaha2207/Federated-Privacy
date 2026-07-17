# -*- coding: utf-8 -*-
"""
============================================================
  MACHINE 2 — IoT Health Monitoring Partition 2
  Model  : SGDClassifier (Logistic Regression)
  Target : Health Event (0, 1, 2, 3)
============================================================
"""

import os
import numpy as np
import pandas as pd
from sklearn.linear_model import SGDClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split

# ── CONFIG ─────────────────────────────────────────────────
DATA_FILE   = os.path.join(os.path.dirname(__file__), "machine2_data.json")
ALPHA       = 0.01   # Regularization parameter
SEED        = 42

class Machine2:
    def __init__(self, epsilon=0.5):
        self.epsilon = epsilon
        self.alpha   = ALPHA
        
        # Initialize Logistic Regression with SGD (L2 penalty)
        self.model = SGDClassifier(
            loss='log_loss', penalty='l2', alpha=self.alpha,
            fit_intercept=True, warm_start=True, random_state=SEED
        )
        
        self.le = LabelEncoder()
        self.le.fit([0, 1, 2, 3])
        
        self.X_train = None
        self.X_test  = None
        self.y_train = None
        self.y_test  = None
        self.n_samples = 0
        
        # Load and prepare data
        self.load_data()
        self.initialize_model()

    def load_data(self):
        df = pd.read_json(DATA_FILE, orient='records')
        
        # Extract features and targets
        feature_cols = [
            'heart_rate', 'blood_oxygen', 'blood_pressure_systolic', 'blood_pressure_diastolic', 
            'glucose_level', 'body_temperature', 'respiratory_rate', 'activity_level', 
            'sleep_quality', 'stress_level', 'hrv_sdnn', 'steps_count', 'calories_burned'
        ]
        
        X = df[feature_cols].values
        y = self.le.transform(df["Daily_Health_Condition"].values)
        
        # Train-Test Split (80% train, 20% test)
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, test_size=0.20, random_state=SEED, stratify=y
        )
        self.n_samples = len(self.X_train)

    def initialize_model(self):
        # Call partial_fit once with dummy subset to initialize weights shape
        self.model.partial_fit(self.X_train[:4], [0, 1, 2, 3], classes=[0, 1, 2, 3])
        # Reset initialized weights to zero
        self.model.coef_ = np.zeros_like(self.model.coef_)
        self.model.intercept_ = np.zeros_like(self.model.intercept_)

    def get_weights(self):
        """Returns local model weights (coefficients & intercept)."""
        return self.model.coef_.copy(), self.model.intercept_.copy()

    def set_weights(self, coef, intercept):
        """Updates local model weights with global weights."""
        self.model.coef_ = coef.copy()
        self.model.intercept_ = intercept.copy()

    def local_train(self, epochs=5):
        """Trains local model for a few epochs on local training set."""
        for _ in range(epochs):
            self.model.partial_fit(self.X_train, self.y_train)

    def get_dp_weights(self):
        """
        Applies Laplace Output Perturbation DP to local weights.
        
        Sensitivity Formula:
            L2 Sensitivity ΔW = 2 / (N * alpha)
            where N = local sample size, alpha = L2 regularization strength.
            
        Laplace Scale:
            b = ΔW / epsilon
        """
        coef, intercept = self.get_weights()
        
        # Calculate sensitivity and noise scale
        sensitivity = 2.0 / (self.n_samples * self.alpha)
        scale = sensitivity / self.epsilon
        
        # Generate and add Laplace noise
        noise_coef = np.random.laplace(loc=0.0, scale=scale, size=coef.shape)
        noise_intercept = np.random.laplace(loc=0.0, scale=scale, size=intercept.shape)
        
        noisy_coef = coef + noise_coef
        noisy_intercept = intercept + noise_intercept
        
        return noisy_coef, noisy_intercept

    def get_test_data(self):
        return self.X_test, self.y_test

    def get_train_size(self):
        return self.n_samples

    def get_dp_info(self):
        sensitivity = 2.0 / (self.n_samples * self.alpha)
        scale = sensitivity / self.epsilon
        return f"Output Perturbation DP | Sensitivity={sensitivity:.6f} | Scale={scale:.6f}"

# ── STANDALONE TEST ────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    from sklearn.metrics import accuracy_score
    
    parser = argparse.ArgumentParser(description="Standalone Client Model with Differential Privacy")
    parser.add_argument("--no-dp", action="store_true", help="Run without Differential Privacy")
    parser.add_argument("-e", "--epsilon", type=float, default=0.5, help="Privacy Budget (Epsilon) [default: 0.5]")
    args = parser.parse_args()
    
    if not args.no_dp and args.epsilon <= 0:
        print("Error: Epsilon must be strictly greater than 0.")
        exit(1)
        
    # Initialize machine
    m = Machine2(epsilon=args.epsilon)
    print(f"\nMachine 2 Standalone Run")
    print(f"Dataset Size: {m.get_train_size()} training samples")
    
    if args.no_dp:
        print("Differential Privacy: Disabled")
    else:
        print(f"Differential Privacy: Enabled (Epsilon = {args.epsilon})")
        print(f"DP Info: {m.get_dp_info()}")
        
    print("Training local model...")
    m.local_train(epochs=15)
    
    # Get weights
    if args.no_dp:
        coef, intercept = m.get_weights()
    else:
        coef, intercept = m.get_dp_weights()
        # Set the noisy weights back to the model for evaluation
        m.set_weights(coef, intercept)
        
    # Evaluate
    X_te, y_te = m.get_test_data()
    preds = m.model.predict(X_te)
    accuracy = accuracy_score(y_te, preds)
    print(f"Test Accuracy: {accuracy*100:.2f}%\n")
