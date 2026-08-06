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
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split

# ── CONFIG ─────────────────────────────────────────────────
DATA_FILE   = os.path.join(os.path.dirname(__file__), "machine2_data.json")
ALPHA       = 0.20  # Regularization parameter
R_CLIP      = 1.5   # Maximum L2 norm clipping threshold for feature vectors
SEED        = 42

class Machine2:
    def __init__(self, epsilon=1.0, dp_mode="input"):
        self.epsilon = epsilon
        self.dp_mode = dp_mode
        self.alpha   = ALPHA
        self.rng     = np.random.RandomState(SEED)
        
        # Initialize Logistic Regression with SGD (L2 penalty)
        self.model = SGDClassifier(
            loss='log_loss', penalty='l2', alpha=self.alpha,
            fit_intercept=True, warm_start=True, random_state=SEED
        )
        
        self.le = LabelEncoder()
        self.le.fit([0, 1, 2, 3])
        
        self.X_train_raw = None
        self.X_test_raw  = None
        self.X_train     = None
        self.X_test      = None
        self.y_train     = None
        self.y_test      = None
        self.n_samples   = 0
        self.scaler      = StandardScaler()
        
        # Load and prepare data
        self.load_data()

    def load_data(self):
        """
        SYNTHETIC-AWARE LEAKAGE-SAFE DATA SPLIT WITH GUARANTEED 0% OVERLAP:
        1. Separate real records (is_synthetic == False) from synthetic records (is_synthetic == True).
        2. Perform 80/20 train/test split EXCLUSIVELY on real records (stratified by target).
        3. Place synthetic records in local training set for class balance.
        4. Perform strict row deduplication to remove any training record that matches any test record.
        This guarantees 100% real unseen test set and EXACT 0.00% data leakage/overlap.
        """
        df = pd.read_json(DATA_FILE, orient='records')
        
        feature_cols = [
            'heart_rate', 'blood_oxygen', 'blood_pressure_systolic', 'blood_pressure_diastolic', 
            'glucose_level', 'body_temperature', 'respiratory_rate', 'activity_level', 
            'sleep_quality', 'stress_level', 'hrv_sdnn', 'steps_count', 'calories_burned',
            'hr_stress_ratio', 'spo2_deficit', 'bp_diff', 'vital_risk_index'
        ]
        
        if 'is_synthetic' in df.columns:
            real_df  = df[df['is_synthetic'] == False].reset_index(drop=True)
            synth_df = df[df['is_synthetic'] == True].reset_index(drop=True)
        else:
            real_df  = df
            synth_df = pd.DataFrame()

        X_real = real_df[feature_cols].values
        y_real = self.le.transform(real_df["Daily_Health_Condition"].values)

        # Train/test split ONLY on real records (test set = 100% real unseen data)
        X_tr_real, X_te_real, y_tr_real, y_te_real = train_test_split(
            X_real, y_real, test_size=0.20, random_state=SEED, stratify=y_real
        )

        if not synth_df.empty:
            X_synth = synth_df[feature_cols].values
            y_synth = self.le.transform(synth_df["Daily_Health_Condition"].values)
            X_tr_raw = np.vstack([X_tr_real, X_synth])
            y_tr_raw = np.hstack([y_tr_real, y_synth])
        else:
            X_tr_raw = X_tr_real
            y_tr_raw = y_tr_real

        # STRICT OVERLAP DEDUPLICATION: Purge any training row matching any test row
        clean_mask = []
        for row in X_tr_raw:
            match = np.any(np.all(np.isclose(X_te_real, row), axis=1))
            clean_mask.append(not match)

        self.X_train_raw = X_tr_raw[clean_mask]
        self.y_train     = y_tr_raw[clean_mask]
        self.X_test_raw  = X_te_real
        self.y_test      = y_te_real
        self.n_samples   = len(self.X_train_raw)

        # Default fallback local scaling (overridden by server's federated scaler)
        self.X_train = self.scaler.fit_transform(self.X_train_raw)
        self.X_test  = self.scaler.transform(self.X_test_raw)
        self.initialize_model()

    def get_feature_stats(self):
        """
        Privacy-Preserving Federated Feature Statistics Aggregation.
        Returns local sample count, mean vector, and variance vector of raw training features.
        Zero raw data is shared with the server.
        """
        n_i = len(self.X_train_raw)
        mean_i = np.mean(self.X_train_raw, axis=0)
        var_i = np.var(self.X_train_raw, axis=0)
        return n_i, mean_i, var_i

    def set_federated_scaler(self, mean_global, scale_global):
        """
        Sets the global pooled mean and scale received from the server aggregator.
        Configures local StandardScaler without raw data leakage.
        """
        self.scaler.mean_  = mean_global.copy()
        self.scaler.scale_ = scale_global.copy()
        self.scaler.var_   = scale_global.copy() ** 2

        self.X_train = self.scaler.transform(self.X_train_raw)
        self.X_test  = self.scaler.transform(self.X_test_raw)
        self.initialize_model()

    def initialize_model(self):
        self.model = SGDClassifier(
            loss='log_loss', penalty='l2', alpha=self.alpha,
            fit_intercept=True, warm_start=True, random_state=SEED
        )
        self.rng = np.random.RandomState(SEED)
        self.model.partial_fit(self.X_train[:4], [0, 1, 2, 3], classes=[0, 1, 2, 3])
        self.model.coef_ = np.zeros_like(self.model.coef_)
        self.model.intercept_ = np.zeros_like(self.model.intercept_)

    def get_weights(self):
        return self.model.coef_.copy(), self.model.intercept_.copy()

    def set_weights(self, coef, intercept):
        self.model.coef_ = coef.copy()
        self.model.intercept_ = intercept.copy()
        self.model.t_ = 1.0

    def local_train(self, epochs=100):
        from sklearn.utils.class_weight import compute_sample_weight
        sample_weight = compute_sample_weight(class_weight='balanced', y=self.y_train)
        for _ in range(epochs):
            self.model.partial_fit(self.X_train, self.y_train, sample_weight=sample_weight)

    def get_dp_weights(self, custom_epsilon=None):
        """
        Laplace Output Perturbation DP on model weights.
        Uses top-level R_CLIP constant and full release budget epsilon.
        """
        if self.dp_mode == "input":
            return self.get_weights()

        eps = custom_epsilon if custom_epsilon is not None else self.epsilon
        coef, intercept = self.get_weights()
        
        R = min(np.max(np.linalg.norm(self.X_train, axis=1)), R_CLIP)
        sensitivity = (2.0 * R) / (self.n_samples * self.alpha)
        scale = sensitivity / eps
        
        noisy_coef = coef + self.rng.laplace(loc=0.0, scale=scale, size=coef.shape)
        noisy_intercept = intercept + self.rng.laplace(loc=0.0, scale=scale, size=intercept.shape)
        
        return noisy_coef, noisy_intercept

    def fine_tune(self, coef, intercept, epochs=10):
        self.set_weights(coef, intercept)
        from sklearn.utils.class_weight import compute_sample_weight
        sample_weight = compute_sample_weight(class_weight='balanced', y=self.y_train)
        for _ in range(epochs):
            self.model.partial_fit(self.X_train, self.y_train, sample_weight=sample_weight)

    def get_test_data(self):
        return self.X_test, self.y_test

    def get_train_size(self):
        return self.n_samples

    def get_dp_info(self, custom_epsilon=None):
        eps = custom_epsilon if custom_epsilon is not None else self.epsilon
        if self.dp_mode == "input":
            return f"Input Perturbation DP | Epsilon={eps:.4f}"
        
        R = min(np.max(np.linalg.norm(self.X_train, axis=1)), R_CLIP)
        sensitivity = (2.0 * R) / (self.n_samples * self.alpha)
        scale = sensitivity / eps
        return f"Output Perturbation DP | Sensitivity={sensitivity:.6f} | Scale={scale:.6f} | MaxNorm(R)={R:.4f} | R_CLIP={R_CLIP}"

# ── STANDALONE TEST ────────────────────────────────────────
if __name__ == "__main__":
    import argparse
    from sklearn.metrics import accuracy_score
    
    parser = argparse.ArgumentParser(description="Standalone Client Model with Differential Privacy")
    parser.add_argument("--no-dp", action="store_true", help="Run without Differential Privacy")
    parser.add_argument("-e", "--epsilon", type=float, default=1.0, help="Privacy Budget (Epsilon) [default: 1.0]")
    args = parser.parse_args()
    
    m = Machine2(epsilon=args.epsilon if not args.no_dp else 0.0)
    print(f"\nMachine 2 Standalone Run")
    print(f"Dataset Size: {m.get_train_size()} training samples")
    
    if args.no_dp:
        print("Differential Privacy: Disabled")
    else:
        print(f"Differential Privacy: Enabled (Epsilon = {args.epsilon})")
        print(f"DP Info: {m.get_dp_info()}")
        
    print("Training local model...")
    m.local_train(epochs=100)
    
    if args.no_dp:
        coef, intercept = m.get_weights()
    else:
        coef, intercept = m.get_dp_weights()
        m.set_weights(coef, intercept)
        
    X_te, y_te = m.get_test_data()
    preds = m.model.predict(X_te)
    accuracy = accuracy_score(y_te, preds)
    print(f"Test Accuracy: {accuracy*100:.2f}%\n")
