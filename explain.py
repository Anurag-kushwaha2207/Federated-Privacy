# -*- coding: utf-8 -*-
"""
Per-Patient Local Explanation Module (On-Device XAI)

PRIVACY BOUNDARY GUARANTEE:
This module executes strictly on-device at the client node.
It computes local SHAP explanations using ONLY the local model and local background data.
No raw patient data (X_patient), training data (X_train_local), or raw SHAP matrices are 
transmitted to the central server. Only lightweight text summaries (e.g. "heart_rate increased risk")
are combined into alerts locally.
"""

import shap
import numpy as np

def explain_prediction(local_model, X_train_local, X_patient, feature_names):
    """
    On-device SHAP explanation for a single patient prediction.
    
    Privacy note:
    This runs ENTIRELY locally, using only this client's own model and own training data 
    as SHAP background. Nothing here is transmitted to the server — same privacy boundary as local_train().
    
    Parameters:
    -----------
    local_model : SGDClassifier
        The locally fine-tuned or local model instance on the client machine.
    X_train_local : np.ndarray
        Local standardized training dataset used as background distribution for LinearExplainer.
    X_patient : np.ndarray
        Single patient feature vector (1D array of length n_features).
    feature_names : list of str
        List of feature column names.
        
    Returns:
    --------
    pred_class : int
        Predicted condition class index (0..3).
    explanation_parts : list of str
        Top-3 feature names and their directional risk contributions.
    """
    # Ensure patient vector is 2D (1, n_features)
    X_p2d = X_patient.reshape(1, -1)
    
    # Initialize linear explainer with local background data
    explainer = shap.LinearExplainer(local_model, X_train_local)
    shap_values = explainer.shap_values(X_p2d)
    
    pred_class = int(local_model.predict(X_p2d)[0])
    
    # Handle multi-class vs binary shap_values format
    if isinstance(shap_values, list):
        contributions = shap_values[pred_class][0]
    elif isinstance(shap_values, np.ndarray):
        if shap_values.ndim == 3:
            if shap_values.shape[0] == 1 and shap_values.shape[2] > pred_class:
                contributions = shap_values[0, :, pred_class]
            else:
                contributions = shap_values[0, pred_class, :]
        elif shap_values.ndim == 2:
            contributions = shap_values[0]
        else:
            contributions = shap_values.ravel()
    else:
        contributions = np.array(shap_values).ravel()
        
    # Rank top-3 contributing features by absolute magnitude
    ranked = sorted(
        zip(feature_names, contributions),
        key=lambda x: -abs(x[1])
    )[:3]
    
    explanation_parts = []
    for fname, val in ranked:
        direction = "increased" if val > 0 else "decreased"
        explanation_parts.append(f"{fname} {direction} risk")
        
    return pred_class, explanation_parts
