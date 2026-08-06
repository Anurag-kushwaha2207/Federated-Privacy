# Privacy-Preserving Federated Learning with Differential Privacy for IoT Health Monitoring

This repository contains the source code and experimental evaluation for a privacy-preserving federated learning system designed for IoT health monitoring devices. The framework allows multiple client edge devices (such as smart health watches) to collaboratively train a global machine learning model for health event classification while keeping patient data private on local devices.

Privacy is guaranteed using **Laplace Output Differential Privacy (DP)** with feature norm clipping and sequential privacy budget composition.

---

## Methodology & Mathematical Formulation

The system operates across decentralized client nodes (Machine 1, Machine 2, Machine 3) with central weight aggregation via Federated Averaging (**FedAvg**).

### 1. Feature Norm Clipping
To bound the maximum impact of any single data point (sensitivity), feature vectors $x_i$ are constrained by an $L_2$ threshold $R_{\text{clip}} = 2.5$:

$$R = \min\left(\max_{i} \|x_i\|_2, R_{\text{clip}}\right)$$

### 2. Sensitivity Calculation ($\Delta W$)
For an $L_2$-regularized convex loss with regularization parameter $\alpha = 0.05$ trained on $N$ local samples:

$$\Delta W = \frac{2 \cdot R}{N \cdot \alpha}$$

### 3. Sequential Epsilon Composition
Given a global privacy budget $\epsilon_{\text{total}} = 1.0$ distributed over $K = 5$ communication rounds, the per-round privacy budget $\epsilon_r$ is:

$$\epsilon_r = \frac{\epsilon_{\text{total}}}{K} = \frac{1.0}{5} = 0.20$$

By the Sequential Composition Theorem:

$$\sum_{r=1}^K \epsilon_r = K \cdot \epsilon_r = 5 \times 0.20 = 1.00$$

### 4. Laplace Noise Injection
Zero-mean Laplace noise scaled by $b = \frac{\Delta W}{\epsilon_r}$ is added to local model coefficients prior to central aggregation:

$$W_{\text{private}} = W_{\text{local}} + \text{Laplace}\left(0, \frac{\Delta W}{\epsilon_r}\right)$$

---

## Experimental Setup & Benchmark Results

The system was evaluated on a clean dataset of 6,000 records (`health_data_balanced_after_overfitting.xlsx`) partitioned among 3 client machines (2,000 samples per client) using a non-IID Dirichlet distribution ($\alpha = 1.0$).

### 1. Federated Learning Benchmarks

| Command | Privacy Configuration | Global Accuracy | Average Personalized Accuracy |
|:---|:---|:---:|:---:|
| `python server.py` | Differential Privacy Enabled ($\epsilon = 1.0$, Default) | **89.25%** | **98.17%** |
| `python server.py --no-dp` | Non-Private Baseline | **96.58%** | **98.17%** |
| `python server.py -e 1.0 --dp-clients 1` | Heterogeneous DP (Client 1 Active) | **96.17%** | **98.17%** |
| `python server.py -e 1.0 --dp-clients 1,2` | Heterogeneous DP (Clients 1 & 2 Active) | **93.08%** | **98.17%** |

### 2. Standalone Client Benchmarks (Local Training Only)

| Command | Client | Differential Privacy | Test Accuracy |
|:---|:---|:---:|:---:|
| `python machine.py --no-dp` | Machine 1 | Disabled | **98.75%** |
| `python machine.py --no-dp` | Machine 2 | Disabled | **97.75%** |
| `python machine.py --no-dp` | Machine 3 | Disabled | **97.50%** |
| `python machine.py` | Machine 1 | Enabled ($\epsilon = 1.0$) | **98.75%** |
| `python machine.py` | Machine 2 | Enabled ($\epsilon = 1.0$) | **97.75%** |
| `python machine.py` | Machine 3 | Enabled ($\epsilon = 1.0$) | **97.50%** |

### 3. Comparison at Budget $\epsilon = 0.5$

| Command | Configuration | Global Accuracy | Average Personalized Accuracy |
|:---|:---|:---:|:---:|
| `python server.py -e 0.5` | All Clients Active DP ($\epsilon = 0.5$) | **57.17%** | **97.75%** |
| `python server.py -e 0.5 --dp-clients 1` | Client 1 Active DP ($\epsilon = 0.5$) | **90.75%** | **97.75%** |
| `python server.py -e 0.5 --dp-clients 1,2` | Clients 1 & 2 Active DP ($\epsilon = 0.5$) | **72.42%** | **97.83%** |
| `python machine.py -e 0.5` | Standalone Clients ($\epsilon = 0.5$) | **98.50% / 96.50% / 96.25%** | — |

---

## Repository Structure

```text
├── README.md                                 # Project documentation and reproduction guide
├── run.txt                                   # Plaintext log of verified experiment results
├── server.py                                 # Central server script for FedAvg aggregation and evaluation
├── split_data.py                             # Data partitioning script implementing non-IID Dirichlet split
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
Run the data splitter to partition the dataset into client files (`machine1_data.json`, `machine2_data.json`, `machine3_data.json`):

```bash
python split_data.py
```

### 2. Run Federated Learning (Default DP $\epsilon = 1.0$)
To run the main federated learning pipeline:

```bash
python server.py
```

### 3. Run Baseline Without Privacy
To evaluate performance without differential privacy noise:

```bash
python server.py --no-dp
```

### 4. Run Custom Privacy Configurations
To test specific privacy budgets or client subsets:

```bash
python server.py -e 0.5
python server.py -e 1.0 --dp-clients 1
```

### 5. Run Standalone Clients
To evaluate local client models without server aggregation:

```bash
python machine.py            # Standalone with DP (epsilon = 1.0)
python machine.py --no-dp    # Standalone without DP
```
