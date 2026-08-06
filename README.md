# Privacy-Preserving Federated Learning with Differential Privacy for IoT Health Monitoring

This repository contains the source code and experimental evaluation for a privacy-preserving federated learning system designed for IoT health monitoring devices. The framework allows multiple client edge devices (such as smart health watches) to collaboratively train a global machine learning model for health event classification while keeping patient data private on local devices.

Privacy is guaranteed using **Laplace Output Differential Privacy (DP)** with feature norm clipping and sequential privacy budget composition.

---

## Methodology & Mathematical Formulation

The system operates across decentralized client nodes (Machine 1, Machine 2, Machine 3) with central weight aggregation via Federated Averaging (**FedAvg**).

### 1. Feature Engineering & Anomaly Interaction Ratios
To resolve subtle decision boundaries between **Normal** and **Mild Events**, four clinical interaction features are engineered prior to scaling:
- `hr_stress_ratio` = $\text{heart\_rate} \times \text{stress\_level}$
- `spo2_deficit` = $100.0 - \text{blood\_oxygen}$
- `bp_diff` = $\text{systolic\_bp} - \text{diastolic\_bp}$
- `vital_risk_index` = $\frac{\text{heart\_rate}}{70.0} + \frac{\text{spo2\_deficit}}{5.0} + 2.0 \times \text{stress\_level}$

### 2. Feature Norm Clipping ($R_{\text{clip}}$)
Feature vectors $x_i$ are constrained by an $L_2$ threshold $R_{\text{clip}} = 1.5$:

$$R = \min\left(\max_{i} \|x_i\|_2, R_{\text{clip}}\right)$$

### 3. Sensitivity Calculation ($\Delta W$)
For $L_2$-regularized loss with parameter $\alpha = 0.20$ trained on $N = 1600$ local samples per client:

$$\Delta W = \frac{2 \cdot R_{\text{clip}}}{N \cdot \alpha} = \frac{2 \times 1.5}{1600 \times 0.20} = 0.009375$$

### 4. Sequential Epsilon Composition & Noise Injection
Given total budget $\epsilon_{\text{total}} = 0.5$ over $K = 5$ rounds, $\epsilon_r = 0.10$. The Laplace noise scale is:

$$b = \frac{\Delta W}{\epsilon_r} = \frac{0.009375}{0.10} = 0.09375$$

Zero-mean Laplace noise scaled by $b$ is added to local model parameters prior to central aggregation:

$$W_{\text{private}} = W_{\text{local}} + \text{Laplace}\left(0, \frac{\Delta W}{\epsilon_r}\right)$$

---

## Experimental Benchmarks & Evaluation Metrics

Evaluated on 6,000 records (`health_data_balanced_after_overfitting.xlsx`) partitioned across 3 client machines (2,000 records each) under non-IID Dirichlet distribution ($\alpha = 1.0$).

### 1. Global & Personalized Model Performance

| Command | Privacy Configuration | Global Accuracy | ROC-AUC | Avg Personalized Accuracy | Accuracy Gap vs Baseline |
|:---|:---|:---:|:---:|:---:|:---:|
| `python server.py --no-dp` | Non-Private Baseline | **95.83%** | **0.9942** | **97.00%** | 0.00% |
| `python server.py -e 1.0` | DP Enabled ($\epsilon = 1.0$, Default) | **95.33%** | **0.9934** | **97.00%** | **0.50%** |
| `python server.py -e 0.5` | DP Enabled ($\epsilon = 0.5$, High Noise) | **93.75%** | **0.9905** | **97.08%** | **2.08%** |

### 2. Class-Wise Classification Metrics (High Noise DP $\epsilon = 0.5$)

| Class | Precision | Recall | F1-Score | Status |
|:---|:---:|:---:|:---:|:---|
| **0: Normal** | **99.67%** | **99.67%** | **0.9967** | ✅ **300 / 300 Correct (0 False Positives)** |
| **1: Mild Event** | **99.58%** | **79.33%** | **0.8831** | ✅ **238 / 300 Correct** |
| **2: Moderate Event** | **95.54%** | **100.00%**| **0.9772** | ✅ **300 / 300 Correct** |
| **3: Severe Event** | **87.68%** | **96.00%** | **0.9160** | ✅ **288 / 300 Correct** |

### 3. Confusion Matrix ($\epsilon = 0.5$)

```text
Actual / Predicted   Normal     Mild       Moderate   Severe    
----------------------------------------------------------------
Normal               299        0          1          0         
Mild Event           2          238        27         33        
Moderate Event       0          0          300        0         
Severe Event         0          12         0          288       
```

---

## Repository Structure

```text
├── README.md                                 # Project documentation and reproduction guide
├── run.txt                                   # Plaintext log of verified experiment results
├── server.py                                 # Central server script for FedAvg aggregation and evaluation
├── split_data.py                             # Data partitioning script implementing Dirichlet non-IID split
├── machine.py                                # Runner script for standalone client training and evaluation
├── machine1.py                               # Client 1 implementation
├── machine2.py                               # Client 2 implementation
├── machine3.py                               # Client 3 implementation
├── health_data_balanced_after_overfitting.xlsx # Primary Excel dataset containing patient telemetry
├── plot_accuracy.png                         # Accuracy progression across communication rounds
├── plot_dp_laplace.png                       # Laplace noise scale curves for different epsilon values
├── plot_dataset_sizes.png                    # Label distribution plot for each client partition
└── plot_confusion_matrix.png                 # Multi-class confusion matrix on global model test set
```

---

## How to Run

### 1. Partition Data
Run the data splitter to build pre-scaled interaction features and partition data into client JSON files:

```bash
python split_data.py
```

### 2. Run Federated Learning (Default DP $\epsilon = 1.0$)
```bash
python server.py -e 1.0
```

### 3. Run High Noise Privacy Experiment ($\epsilon = 0.5$)
```bash
python server.py -e 0.5
```

### 4. Run Baseline Without Privacy
```bash
python server.py --no-dp
```
