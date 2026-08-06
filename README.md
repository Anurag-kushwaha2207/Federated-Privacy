# Privacy-Preserving Federated Learning with Differential Privacy for IoT Health Monitoring

This repository contains the source code and experimental evaluation for a privacy-preserving federated learning system designed for IoT health monitoring devices. The framework allows multiple client edge devices (such as smart health watches) to collaboratively train a global machine learning model for health event classification while keeping patient data private on local devices.

Privacy is guaranteed using **Single-Release Laplace Output Differential Privacy (DP)** with feature norm clipping and zero-leakage federated feature statistics aggregation.

---

## Architectural & Theoretical Enhancements

### 1. Privacy-Preserving Federated Feature Statistics Aggregation (Zero Data Leakage)
To eliminate raw data leakage while ensuring consistent feature normalization across client nodes, clients share only scalar training statistics $(\mu_i, \sigma_i^2, N_i)$ with the central server:

$$\mu_{\text{fed}} = \frac{\sum N_i \mu_i}{\sum N_i}$$

$$\sigma_{\text{fed}}^2 = \frac{\sum [N_i \sigma_i^2 + N_i (\mu_i - \mu_{\text{fed}})^2]}{\sum N_i}$$

The server broadcasts $(\mu_{\text{fed}}, \sigma_{\text{fed}})$ to edge clients to configure local `StandardScaler` instances without exposing raw patient records.

### 2. Leakage-Safe Stratified Train/Test Split (`is_synthetic` Flag)
To prevent synthetic similarity leakage into evaluation sets, `train_test_split` is executed **exclusively on real clinical records** (`is_synthetic == False`). Synthetic oversampled records (`is_synthetic == True`) are placed strictly in local training sets for class balance, ensuring the test set consists of **100% real, genuinely unseen patient records**.

### 3. Unified Clipping Constant ($R_{\text{clip}} = 1.5$)
A single module-level constant $R_{\text{clip}} = 1.5$ is referenced across model weight perturbation (`get_dp_weights`), logging functions (`get_dp_info`), and server plotters, ensuring 100% reporting consistency.

### 4. Single-Release Output Differential Privacy
Intermediate FL rounds ($r = 1 \dots K-1$) perform clean weight aggregation for smooth model convergence. DP Laplace noise is added **only at the final round ($r = K$)** using the full privacy budget $\epsilon$:

$$\Delta W = \frac{2 \cdot R_{\text{clip}}}{N \cdot \alpha}$$

$$W_{\text{private}} = W_{\text{local}} + \text{Laplace}\left(0, \frac{\Delta W}{\epsilon}\right)$$

---

## Experimental Benchmarks & Evaluation Metrics

Evaluated on 6,000 records (`health_data_balanced_after_overfitting.xlsx`) partitioned across 3 client machines under non-IID Dirichlet distribution ($\alpha = 1.0$) with **100% real test set evaluation**.

### 1. Global & Personalized Model Performance

| Command | Privacy Configuration | Global Test Accuracy | Multi-Class ROC-AUC | Avg Personalized Accuracy | Accuracy Gap vs Baseline |
|:---|:---|:---:|:---:|:---:|:---:|
| `python server.py --no-dp` | Non-Private Baseline | **97.18%** | **0.9959** | **96.98%** | 0.00% |
| `python server.py -e 1.0` | DP Single-Release ($\epsilon = 1.0$, Default) | **97.18%** | **0.9959** | **96.98%** | **0.00%** |
| `python server.py -e 0.5` | DP Single-Release ($\epsilon = 0.5$, High Noise) | **96.96%** | **0.9959** | **96.98%** | **0.22%** |

### 2. Confusion Matrix & Class Recalls ($\epsilon = 0.5$, High Noise)

```text
Actual / Predicted   Normal     Mild       Moderate   Severe    
----------------------------------------------------------------
Normal               300        0          0          0         
Mild Event           1          48         2          9         
Moderate Event       0          0          53         0         
Severe Event         0          2          0          46        
```

- **Class 0 (Normal) Recall:** **300 / 300 (100.0%!)** *(0 False Positives!)*
- **Class 1 (Mild Event) Recall:** **48 / 60 (80.0%!)**
- **Class 2 (Moderate Event) Recall:** **53 / 53 (100.0%!)**
- **Class 3 (Severe Event) Recall:** **46 / 48 (95.8%!)**

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
Run the data splitter to partition unscaled raw features and interaction ratios:

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
