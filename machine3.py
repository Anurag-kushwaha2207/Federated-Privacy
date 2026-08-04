# -*- coding: utf-8 -*-
"""
============================================================
  MACHINE 3 — IoT Health Monitoring Partition 3
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
DATA_FILE   = os.path.join(os.path.dirname(__file__), "machine3_data.json")
ALPHA       = 0.1   # Regularization parameter
SEED        = 42

class Machine3:
    def __init__(self, epsilon=0.5, dp_mode="input"):
        self.epsilon = epsilon
        self.dp_mode = dp_mode
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
        
        # Fit local StandardScaler on client training data only to avoid global data leakage
        from sklearn.preprocessing import StandardScaler
        self.scaler = StandardScaler()
        self.X_train = self.scaler.fit_transform(self.X_train)
        self.X_test = self.scaler.transform(self.X_test)
        self.n_samples = len(self.X_train)
        
        # Apply Input Perturbation DP if enabled
        if self.dp_mode == "input" and self.epsilon > 0:
            self.apply_input_dp()

    def initialize_model(self):
        # Recreate model to reset step counter t_ and optimizer state completely
        self.model = SGDClassifier(
            loss='log_loss', penalty='l2', alpha=self.alpha,
            fit_intercept=True, warm_start=True, random_state=SEED
        )
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
        # Reset step counter so local training starts with a fresh learning rate schedule
        self.model.t_ = 1.0

    def local_train(self, epochs=100):
        """
        Trains local model on local training set.
        We train for a large number of epochs (100+) to ensure near-convergence,
        which validates the convergence assumption of Output Perturbation DP sensitivity.
        We compute balanced sample weights to address class imbalance.
        """
        from sklearn.utils.class_weight import compute_sample_weight
        sample_weight = compute_sample_weight(class_weight='balanced', y=self.y_train)
        for _ in range(epochs):
            self.model.partial_fit(self.X_train, self.y_train, sample_weight=sample_weight)

    def classify_feature_sensitivity(self, feature_name):
        """Categorizes features into sensitivity levels (High/Medium/Low)."""
        high_sens = ['glucose_level', 'blood_pressure_systolic', 'blood_pressure_diastolic', 'heart_rate']
        medium_sens = ['body_temperature', 'respiratory_rate', 'sleep_quality', 'stress_level', 'hrv_sdnn']
        low_sens = ['activity_level', 'steps_count', 'calories_burned']
        
        if feature_name in high_sens:
            return "High", 1.5
        elif feature_name in medium_sens:
            return "Medium", 1.0
        else:
            return "Low", 0.1

    def apply_input_dp(self):
        """Applies Laplace Local DP (Input Perturbation) to training features based on sensitivity."""
        feature_cols = [
            'heart_rate', 'blood_oxygen', 'blood_pressure_systolic', 'blood_pressure_diastolic', 
            'glucose_level', 'body_temperature', 'respiratory_rate', 'activity_level', 
            'sleep_quality', 'stress_level', 'hrv_sdnn', 'steps_count', 'calories_burned'
        ]
        # Standardized features typically lie in [-3, 3] range. We assume a clip bound B = 3.
        # Sensitivity of a feature is the width of its range: Delta = 2 * B = 6.0
        Delta = 6.0
        
        for idx, col in enumerate(feature_cols):
            sens_class, multiplier = self.classify_feature_sensitivity(col)
            # Epsilon per feature is scaled by the multiplier
            # High sensitivity = more noise (smaller effective epsilon), Low sensitivity = less noise (larger effective epsilon)
            feat_eps = self.epsilon / multiplier
            scale = Delta / feat_eps
            
            # Add Laplace noise to the feature column
            noise = np.random.laplace(loc=0.0, scale=scale, size=self.X_train[:, idx].shape)
            self.X_train[:, idx] += noise

    def get_dp_weights(self):
        """
        Applies Laplace Output Perturbation DP to local weights if dp_mode is 'output'.
        If dp_mode is 'input', returns clean weights as noise is already applied to input features.
        
        Sensitivity Formula:
            L2 Sensitivity ΔW = 2 * R / (N * alpha)
            where N = local sample size, alpha = L2 regularization strength,
            and R = maximum L2 norm of the input feature vectors.
            
        Laplace Scale:
            b = ΔW / epsilon
            
        NOTE: This output perturbation mechanism assumes the local optimization
        objective has fully converged to the unique L2-regularized ERM minimizer.
        To approximate this condition, local training epochs should be set high.
        """
        if self.dp_mode == "input":
            # Post-processing theorem: model trained on already DP-perturbed inputs inherits DP automatically
            return self.get_weights()

        coef, intercept = self.get_weights()
        
        # Calculate maximum L2 norm of the features
        R = np.max(np.linalg.norm(self.X_train, axis=1))
        
        # Calculate sensitivity and noise scale
        sensitivity = (2.0 * R) / (self.n_samples * self.alpha)
        scale = sensitivity / self.epsilon
        
        # Generate and add Laplace noise
        noise_coef = np.random.laplace(loc=0.0, scale=scale, size=coef.shape)
        noise_intercept = np.random.laplace(loc=0.0, scale=scale, size=intercept.shape)
        
        # The scale parameter specifies the scale parameter of the Laplace distribution.
        noisy_coef = coef + noise_coef
        noisy_intercept = intercept + noise_intercept
        
        return noisy_coef, noisy_intercept

    def fine_tune(self, coef, intercept, epochs=10):
        """
        Initializes the model weights with the global weights and fine-tunes
        the model locally on client training data for personalization.
        We compute balanced sample weights to address class imbalance.
        """
        self.set_weights(coef, intercept)
        from sklearn.utils.class_weight import compute_sample_weight
        sample_weight = compute_sample_weight(class_weight='balanced', y=self.y_train)
        for _ in range(epochs):
            self.model.partial_fit(self.X_train, self.y_train, sample_weight=sample_weight)

    def get_test_data(self):
        return self.X_test, self.y_test

    def get_train_size(self):
        return self.n_samples

    def get_dp_info(self):
        if self.dp_mode == "input":
            return f"Input Perturbation DP (Local DP) | Epsilon={self.epsilon:.4f} | Mode=Feature-Level Sensitivity"
        
        R = np.max(np.linalg.norm(self.X_train, axis=1))
        sensitivity = (2.0 * R) / (self.n_samples * self.alpha)
        scale = sensitivity / self.epsilon
        return f"Output Perturbation DP | Sensitivity={sensitivity:.6f} | Scale={scale:.6f} | MaxNorm(R)={R:.4f}"

# ── STANDALONE TEST ────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    from sklearn.metrics import accuracy_score
    
    parser = argparse.ArgumentParser(description="Standalone Client Model with Differential Privacy")
    parser.add_argument("--no-dp", action="store_true", help="Run without Differential Privacy")
    parser.add_argument("-e", "--epsilon", type=float, default=0.5, help="Privacy Budget (Epsilon) [default: 0.5]")
    parser.add_argument("--dp-mode", type=str, choices=["input", "output"], default="input", help="DP Perturbation stage: 'input' or 'output'")
    args = parser.parse_args()
    
    if not args.no_dp and args.epsilon <= 0:
        print("Error: Epsilon must be strictly greater than 0.")
        exit(1)
        
    # Initialize machine
    m = Machine3(epsilon=args.epsilon if not args.no_dp else 0.0, dp_mode=args.dp_mode)
    print(f"\nMachine 3 Standalone Run")
    print(f"Dataset Size: {m.get_train_size()} training samples")
    
    if args.no_dp:
        print("Differential Privacy: Disabled")
    else:
        print(f"Differential Privacy: Enabled (Epsilon = {args.epsilon}, Mode = {args.dp_mode})")
        print(f"DP Info: {m.get_dp_info()}")
        
    print("Training local model...")
    m.local_train(epochs=100)
    
    # Get weights
    if args.no_dp:
        coef, intercept = m.get_weights()
    else:
        coef, intercept = m.get_dp_weights()
        # Set the weights back to the model for evaluation
        m.set_weights(coef, intercept)
        
    # Evaluate
    X_te, y_te = m.get_test_data()
    preds = m.model.predict(X_te)
    accuracy = accuracy_score(y_te, preds)
    print(f"Test Accuracy: {accuracy*100:.2f}%\n")
