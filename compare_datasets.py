import pandas as pd

m1 = pd.read_json('machine1_data.json', orient='records')
m2 = pd.read_json('machine2_data.json', orient='records')
m3 = pd.read_json('machine3_data.json', orient='records')

print('='*60)
print('  DATASET COMPARISON REPORT')
print('='*60)

# 1. Dataset Sizes
print()
print('--- 1. SIZE ---')
print(f'Machine 1: {len(m1)} rows x {len(m1.columns)} cols')
print(f'Machine 2: {len(m2)} rows x {len(m2.columns)} cols')
print(f'Machine 3: {len(m3)} rows x {len(m3.columns)} cols')

# 2. Check Column Alignment
print()
print('--- 2. COLUMN EQUALITY ---')
c1 = set(m1.columns)
c2 = set(m2.columns)
c3 = set(m3.columns)
print(f'M1 == M2 columns: {c1 == c2}')
print(f'M1 == M3 columns: {c1 == c3}')
print(f'M2 == M3 columns: {c2 == c3}')

# 3. Common Columns
common_all = c1 & c2 & c3
print()
print('--- 3. COMMON COLUMNS ---')
print(f'Common across all three: {common_all}')
print(f'M1 & M2 common  : {c1 & c2 - common_all}')
print(f'M1 & M3 common  : {c1 & c3 - common_all}')
print(f'M2 & M3 common  : {c2 & c3 - common_all}')

# 4. Age Distribution
print()
print('--- 4. AGE DISTRIBUTION ---')
a1 = pd.to_numeric(m1['Column1.Age'], errors='coerce')
a2 = pd.to_numeric(m2['Age'], errors='coerce')
a3 = pd.to_numeric(m3['Age'], errors='coerce')
print(f'M1 Age: min={int(a1.min())} max={int(a1.max())} avg={a1.mean():.1f}')
print(f'M2 Age: min={int(a2.min())} max={int(a2.max())} avg={a2.mean():.1f}')
print(f'M3 Age: min={int(a3.min())} max={int(a3.max())} avg={a3.mean():.1f}')

# 5. Patient Name Overlap
print()
print('--- 5. PATIENT NAME OVERLAP ---')
m1_names = set(m1['Column1.Name'].dropna().astype(str).str.strip().str.lower())
m2_names = set(m2['Patient_Name'].dropna().astype(str).str.strip().str.lower())
m3_names = set(m3['Name'].dropna().astype(str).str.strip().str.lower())
m1m2 = m1_names & m2_names
m1m3 = m1_names & m3_names
m2m3 = m2_names & m3_names
print(f'M1 unique names: {len(m1_names)}')
print(f'M2 unique names: {len(m2_names)}')
print(f'M3 unique names: {len(m3_names)}')
print(f'M1 & M2 same names: {len(m1m2)} -> {list(m1m2)[:3] if m1m2 else "None"}')
print(f'M1 & M3 same names: {len(m1m3)} -> {list(m1m3)[:3] if m1m3 else "None"}')
print(f'M2 & M3 same names: {len(m2m3)} -> {list(m2m3)[:3] if m2m3 else "None"}')

# 6. Target Variable Distribution
print()
print('--- 6. TARGET VARIABLE ---')
print('M1 Health Condition:')
print(m1['Column1.Health Condition'].value_counts().to_string())
print()
print('M2 Daily_Health_Condition:')
print(m2['Daily_Health_Condition'].value_counts().to_string())
print()
print('M3 Health_Condition:')
print(m3['Health_Condition'].value_counts().to_string())

# 7. Data Domain
print()
print('--- 7. DATA DOMAIN ---')
print('M1: Lifestyle Survey Data (Diet, Sleep, Smoking, Occupation...)')
print('M2: Clinical Daily Logs (Doctor, Medication, Symptoms...)')
print('M3: IoT Wearable Sensor (Steps, Heart Rate, BMI, Calories...)')

print()
print('='*60)
if len(m1m2) == 0 and len(m1m3) == 0 and len(m2m3) == 0:
    print('RESULT: All three datasets are distinct.')
    print('  - Separate columns, domains, and patient cohorts')
    print('  - No overlapping patient records found')
else:
    print(f'RESULT: Name overlap detected across partitions.')
print('='*60)
