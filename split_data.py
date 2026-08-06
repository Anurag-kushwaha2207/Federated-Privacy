import pandas as pd
import numpy as np
import os

base_dir = os.path.dirname(os.path.abspath(__file__))
excel_path = os.path.join(base_dir, "health_data_balanced_after_overfitting.xlsx")

print("Loading dataset from Excel path...")
df = pd.read_excel(excel_path, sheet_name="Balanced_Data")

# Feature Engineering: Add non-linear vital anomaly interaction features
df['hr_stress_ratio'] = df['heart_rate'] * df['stress_level']
df['spo2_deficit'] = 100.0 - df['blood_oxygen']
df['bp_diff'] = df['blood_pressure_systolic'] - df['blood_pressure_diastolic']
df['vital_risk_index'] = (df['heart_rate'] / 70.0) + (df['spo2_deficit'] / 5.0) + (df['stress_level'] * 2.0)

# Features we want to keep
feature_cols = [
    'heart_rate', 'blood_oxygen', 'blood_pressure_systolic', 'blood_pressure_diastolic', 
    'glucose_level', 'body_temperature', 'respiratory_rate', 'activity_level', 
    'sleep_quality', 'stress_level', 'hrv_sdnn', 'steps_count', 'calories_burned',
    'hr_stress_ratio', 'spo2_deficit', 'bp_diff', 'vital_risk_index'
]

df = df.dropna(subset=feature_cols + ['health_event']).reset_index(drop=True)

X = df[feature_cols].copy()
y = df['health_event'].astype(int).copy()

# Fit global StandardScaler so all client partitions share identical, aligned feature scales
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Build a clean DataFrame with scaled features
clean_df = pd.DataFrame(X_scaled, columns=feature_cols)
clean_df['Daily_Health_Condition'] = y  # Keep target name consistent for clients

def partition_data_non_iid_equal_sizes(df, n_clients=3, alpha=0.5, random_seed=42):
    """
    Partitions the dataset among clients using a Dirichlet distribution for non-IID distribution,
    while guaranteeing that each machine receives an equal dataset size (1700, 1700, 1694).
    """
    np.random.seed(random_seed)
    n_classes = 4
    total_samples = len(df)
    
    # Target sizes to keep dataset sizes equal across machines
    base_size = total_samples // n_clients
    target_sizes = [base_size] * (n_clients - 1) + [total_samples - base_size * (n_clients - 1)]
    client_indices = [[] for _ in range(n_clients)]
    
    # Enforce minimum samples per class per client to prevent degenerate models
    min_samples_per_class = 300
    remaining_capacity = list(target_sizes)
    
    # 1. Allocate minimum representation of each class to each client
    for c in range(n_classes):
        class_indices = df[df['Daily_Health_Condition'] == c].index.values.copy()
        np.random.shuffle(class_indices)
        
        for client_id in range(n_clients):
            start_idx = client_id * min_samples_per_class
            end_idx = (client_id + 1) * min_samples_per_class
            client_indices[client_id].extend(class_indices[start_idx:end_idx])
            remaining_capacity[client_id] -= min_samples_per_class
            
    # 2. Gather remaining indices for each class
    remaining_by_class = {}
    for c in range(n_classes):
        class_indices = df[df['Daily_Health_Condition'] == c].index.values.copy()
        np.random.shuffle(class_indices)
        remaining_by_class[c] = list(class_indices[n_clients * min_samples_per_class:])
        
    # 3. Distribute remaining indices using Dirichlet proportions, capped by client capacity
    for c in range(n_classes):
        rem_indices = remaining_by_class[c]
        if not rem_indices:
            continue
            
        proportions = np.random.dirichlet([alpha] * n_clients)
        raw_counts = (proportions * len(rem_indices)).astype(int)
        diff = len(rem_indices) - sum(raw_counts)
        for i in range(diff):
            raw_counts[i % n_clients] += 1
            
        idx = 0
        for client_id in range(n_clients):
            count_to_assign = raw_counts[client_id]
            actual_count = min(count_to_assign, remaining_capacity[client_id])
            
            client_indices[client_id].extend(rem_indices[idx: idx + actual_count])
            remaining_capacity[client_id] -= actual_count
            idx += actual_count
            
        # Distribute any leftover due to capacity capping
        leftover_indices = rem_indices[idx:]
        for index in leftover_indices:
            assigned = False
            for client_id in range(n_clients):
                if remaining_capacity[client_id] > 0:
                    client_indices[client_id].append(index)
                    remaining_capacity[client_id] -= 1
                    assigned = True
                    break
            if not assigned:
                client_indices[np.argmin(remaining_capacity)].append(index)
                
    # Return split DataFrames
    client_dfs = []
    for client_id in range(n_clients):
        indices = client_indices[client_id]
        np.random.shuffle(indices)
        client_dfs.append(df.iloc[indices].reset_index(drop=True))
        
    return client_dfs

print("Partitioning dataset using Dirichlet non-IID split with equal sizes (alpha=1.0)...")
m1_data, m2_data, m3_data = partition_data_non_iid_equal_sizes(clean_df, n_clients=3, alpha=1.0)

print(f"\nPartitions:")
for i, m_data in enumerate([m1_data, m2_data, m3_data], 1):
    print(f"  Machine {i}: {len(m_data)} records")
    dist = m_data['Daily_Health_Condition'].value_counts().sort_index().to_dict()
    print(f"    Label Distribution: {dist}")

# Save as JSON files
m1_path = os.path.join(base_dir, "machine1_data.json")
m2_path = os.path.join(base_dir, "machine2_data.json")
m3_path = os.path.join(base_dir, "machine3_data.json")

m1_data.to_json(m1_path, orient="records", indent=4)
m2_data.to_json(m2_path, orient="records", indent=4)
m3_data.to_json(m3_path, orient="records", indent=4)

print("\nData splitting complete and raw partitions saved successfully (Dirichlet Non-IID Partitioning)!")
