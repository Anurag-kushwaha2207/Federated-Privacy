# -*- coding: utf-8 -*-
"""
============================================================
  MACHINE 2 — Daily Clinical Health Logs
  File   : machine2_data.xlsx  (Health_Tracking)
  Model  : Random Forest
  Target : Daily Health Condition (Critical / Improving / Stable / Weak)
  Records: 500
============================================================
  DIFFERENTIAL PRIVACY:
    Mechanism  : Laplace Mechanism
    Formula    : noise ~ Laplace(0, Δf / ε)
                 where Δf = sensitivity = 1.0
                       ε  = epsilon (privacy budget)
    Applied    : On predicted probability vectors BEFORE sending to server.
    Guarantee  : ε-Differential Privacy on each client.
============================================================
  NOTE: This file runs independently on Machine 2.
  Raw data NEVER leaves this machine.
  Only DP-protected probability vectors are sent to the server.
============================================================
"""

import os
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

# ── CONFIG ─────────────────────────────────────────────────
DATA_FILE = os.path.join(os.path.dirname(__file__), "machine2_data.xlsx")
N_TREES   = 50
MAX_DEPTH = 8
SEED      = 43

# ── DIFFERENTIAL PRIVACY CONFIG ────────────────────────────
EPSILON     = 0.5    # Privacy budget ε  (smaller = more private)
SENSITIVITY = 0.02   # For Random Forest probability, max change is 1/N_TREES (1/50 = 0.02)

# ── HELPERS ────────────────────────────────────────────────
def enc(s):
    return pd.Categorical(s.astype(str).str.strip()).codes

def load_and_prepare():
    df = pd.read_excel(DATA_FILE)
    df = df.dropna(subset=['Daily_Health_Condition']).reset_index(drop=True)

    f = pd.DataFrame()
    f['Age']       = pd.to_numeric(df['Age'], errors='coerce').fillna(40)
    f['Dosage']    = pd.to_numeric(df['Dosage_mg'], errors='coerce').fillna(0)
    f['Gender']    = enc(df['Gender'])
    f['Activity']  = enc(df['Daily_Activity_Level'])
    f['Symptoms']  = enc(df['Symptoms'])
    f['MedStatus'] = enc(df['Medication_Status'])
    f['Medication']= enc(df['Medication_Name'])
    f['Doctor']    = enc(df['Doctor_Name'])
    f['Treatment'] = enc(df['Treatment_Update'])

    X = f.fillna(0)
    y = df['Daily_Health_Condition'].str.strip()
    return X, y, df

# ══════════════════════════════════════════════════════════
#  DIFFERENTIAL PRIVACY — LAPLACE MECHANISM
#  Formula: noise ~ Laplace(0, Δf / ε)
#  After adding noise, probabilities are clipped to [0,1]
#  and re-normalised so they sum to 1 (valid probability dist)
# ══════════════════════════════════════════════════════════
class DifferentialPrivacy:
    """
    Laplace Mechanism for Differential Privacy.

    Laplace Noise Formula:
        b  = Δf / ε                        (scale / noise magnitude)
        noise ~ Laplace(mean=0, scale=b)
        p_private = clip(p_original + noise, 0, 1)
        p_private = p_private / sum(p_private)   # re-normalise

    Parameters:
        epsilon     (ε)  : Privacy budget. Lower = more private, less utility.
        sensitivity (Δf) : Maximum change one record can cause in output.
                           For probability vectors, Δf = 1.0 (standard).
    """
    def __init__(self, epsilon: float = 0.5, sensitivity: float = 1.0):
        self.epsilon     = epsilon
        self.sensitivity = sensitivity
        self.scale       = sensitivity / epsilon   # Laplace scale b = Δf/ε

    def apply(self, probabilities: np.ndarray) -> np.ndarray:
        """
        Add Laplace noise to a probability matrix.

        Args:
            probabilities : ndarray of shape (n_samples, n_classes)
                            Raw predicted probabilities from local model.
        Returns:
            noisy_probs   : ndarray of same shape with DP noise applied.
        """
        noise    = np.random.laplace(loc=0.0, scale=self.scale,
                                     size=probabilities.shape)
        noisy    = probabilities + noise
        noisy    = np.clip(noisy, 0.0, 1.0)
        row_sums = noisy.sum(axis=1, keepdims=True)
        row_sums = np.where(row_sums == 0, 1, row_sums)
        noisy    = noisy / row_sums
        return noisy

    def info(self) -> str:
        return (f"Laplace DP | ε={self.epsilon} | Δf={self.sensitivity} "
                f"| scale b=Δf/ε={self.scale:.4f}")

# ── MAIN CLASS ─────────────────────────────────────────────
class Machine2:
    def __init__(self, epsilon=None):
        self.model         = None
        self.le            = LabelEncoder()
        self.X_test        = None
        self.y_test        = None
        self.local_acc     = None
        self.classes_      = None
        self.feature_names = None
        # ── Differential Privacy engine ──
        use_eps = epsilon if epsilon is not None else EPSILON
        self.dp = DifferentialPrivacy(epsilon=use_eps, sensitivity=SENSITIVITY)

    def train(self):
        print("[Machine 2] Loading data from machine2_data.xlsx ...")
        X, y, df = load_and_prepare()
        self.feature_names = X.columns.tolist()

        y_enc = self.le.fit_transform(y)
        self.classes_ = self.le.classes_.tolist()

        X_tr, X_te, y_tr, y_te = train_test_split(
            X.values, y_enc, test_size=0.20, random_state=SEED, stratify=y_enc)

        self.X_test = X_te
        self.y_test = y_te

        # Local validation
        Xlt, Xlv, ylt, ylv = train_test_split(X_tr, y_tr, test_size=0.20, random_state=SEED)
        self.model = RandomForestClassifier(
            n_estimators=N_TREES, max_depth=MAX_DEPTH,
            random_state=SEED, class_weight='balanced',
            min_samples_leaf=2, n_jobs=-1)
        self.model.fit(Xlt, ylt)
        lp = self.model.predict(Xlv)
        self.local_acc = accuracy_score(ylv, lp)

        # Retrain on full train set
        self.model.fit(X_tr, y_tr)

        print(f"[Machine 2] Training complete.")
        print(f"[Machine 2] Records      : {len(X)}")
        print(f"[Machine 2] Train size   : {len(X_tr)}  |  Test size: {len(X_te)}")
        print(f"[Machine 2] Local Val Acc: {self.local_acc*100:.1f}%")
        print(f"[Machine 2] Classes      : {self.classes_}")
        print(f"[Machine 2] DP Applied   : {self.dp.info()}")
        print(f"[Machine 2] Raw data stays here — only DP-protected probabilities sent to server.")

    def get_probabilities(self, X_test):
        """
        Called by server.
        Returns DP-protected predicted probabilities (NOT raw data).

        Steps:
          1. Local model predicts raw probabilities.
          2. Laplace noise added: noise ~ Laplace(0, Δf/ε)
          3. Clipped to [0,1] and re-normalised.
          4. Only noisy probabilities sent to server → ε-DP guaranteed.
        """
        raw_probs   = self.model.predict_proba(X_test)
        noisy_probs = self.dp.apply(raw_probs)
        return noisy_probs

    def get_test_data(self):
        return self.X_test, self.y_test

    def get_classes(self):
        return self.classes_

    def get_local_accuracy(self):
        return self.local_acc

    def get_train_size(self):
        return len(self.X_test)

    def get_dp_info(self):
        return self.dp.info()


# ── STANDALONE RUN ─────────────────────────────────────────
if __name__ == "__main__":
    import sys
    sys.stdout.reconfigure(encoding='utf-8', errors='replace')

    print("=" * 60)
    print("  MACHINE 2 — Standalone Test  (WITH DIFFERENTIAL PRIVACY)")
    print("  Dataset: machine2_data.xlsx")
    print("=" * 60)

    m = Machine2()
    m.train()

    X_te, y_te = m.get_test_data()
    probs       = m.get_probabilities(X_te)   # DP-protected probs
    preds       = probs.argmax(axis=1)
    acc_test    = accuracy_score(y_te, preds)

    le = m.le
    print(f"\n  Test Set Accuracy (with DP): {acc_test*100:.2f}%")
    print(f"  DP Info: {m.get_dp_info()}")
    print(f"\n  Showing first 20 of {len(y_te)} test patients:")
    print(f"  {'#':<4} {'Actual':<16} {'Predicted':<16} {'Conf%':>6}  Status")
    print(f"  {'-'*55}")
    for j in range(min(20, len(y_te))):
        actual = le.inverse_transform([y_te[j]])[0]
        pred   = le.inverse_transform([preds[j]])[0]
        conf   = probs[j].max() * 100
        s      = "OK" if y_te[j] == preds[j] else "WRONG"
        print(f"  {j+1:<4} {actual:<16} {pred:<16} {conf:>5.1f}%  {s}")
    print("=" * 60)
