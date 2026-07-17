import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import StandardScaler

base_dir = os.path.dirname(os.path.abspath(__file__))
csv_path = r"G:\My Drive\🔬 BrijeshSir_Lab_Work\Kaggel_dataSet\iot_health_monitoring_dataset.csv"

print("Loading dataset from CSV path...")
df = pd.read_csv(csv_path)

# Features we want to keep
feature_cols = [
    'heart_rate', 'blood_oxygen', 'blood_pressure_systolic', 'blood_pressure_diastolic', 
    'glucose_level', 'body_temperature', 'respiratory_rate', 'activity_level', 
    'sleep_quality', 'stress_level', 'hrv_sdnn', 'steps_count', 'calories_burned'
]

X = df[feature_cols].copy()
y = df['health_event'].copy()

# 1. Scale features globally for logistic regression convergence
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Build a clean DataFrame
clean_df = pd.DataFrame(X_scaled, columns=feature_cols)
clean_df['Daily_Health_Condition'] = y  # Keep target name consistent for clients

# Shuffle with seed 42
df_shuffled = clean_df.sample(frac=1.0, random_state=42).reset_index(drop=True)

# Split into 3 partitions: 1700, 1700, 1694
m1_data = df_shuffled.iloc[0:1700].reset_index(drop=True)
m2_data = df_shuffled.iloc[1700:3400].reset_index(drop=True)
m3_data = df_shuffled.iloc[3400:5094].reset_index(drop=True)

print(f"\nPartitions:")
print(f"  Machine 1: {len(m1_data)} records")
print(f"  Machine 2: {len(m2_data)} records")
print(f"  Machine 3: {len(m3_data)} records")

# Save as JSON files
m1_path = os.path.join(base_dir, "machine1_data.json")
m2_path = os.path.join(base_dir, "machine2_data.json")
m3_path = os.path.join(base_dir, "machine3_data.json")

m1_data.to_json(m1_path, orient="records", indent=4)
m2_data.to_json(m2_path, orient="records", indent=4)
m3_data.to_json(m3_path, orient="records", indent=4)

print("\nData splitting and scaling complete and saved successfully!")
