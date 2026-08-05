# Privacy-Preserving IoT Health Monitoring System via Federated Learning & Differential Privacy

An end-to-end, mathematically rigorous **Privacy-Preserving Federated Learning (FL)** framework integrated with **Laplace Output Differential Privacy (DP)** for multi-client IoT health monitoring and predictive health event classification.

---

## 📌 Project Overview

This project implements a decentralized health monitoring system where multi-client edge nodes (Machine 1, Machine 2, Machine 3) perform local model training on sensitive vital signs (heart rate, blood oxygen, blood pressure, glucose, temperature, etc.) without sharing raw medical data with a central server.

* **Federated Learning Algorithm:** Federated Averaging (**FedAvg**) across $K=5$ communication rounds.
* **Privacy Mechanism:** **Laplace Output Perturbation Differential Privacy** with feature norm clipping ($R_{\text{clip}} = 5.0$), exact sensitivity calibration, and sequential epsilon composition ($K \times \epsilon_r = \epsilon_{\text{total}}$).
* **Dataset Partitioning:** 20,721 clean health records partitioned using Non-IID Dirichlet distribution ($\alpha=1.0$) into equal sizes of **6,907 records per client**.
* **Personalization:** Client-side local fine-tuning after global model aggregation.
* **Clinical Risk Mapping:** Automatic conversion of model predictions into **Low, Medium, and High** risk level recommendations and emergency alerts.

---

## 📊 Benchmark Execution & Accuracy Results (`run.txt`)

The following benchmark numbers represent the verified, non-overfitted performance across all client and server configurations:

```text
*****************WITH FEDERATED LEARNING (OUTPUT DP - DEFAULT EPSILON = 1.0)******************
python server.py                                 -> 92.52% (Avg Personalized: 96.41%)  ------------All machine DP 1.0 (DEFAULT)
python server.py --no-dp                         -> 94.55% (Avg Personalized: 96.41%)
python server.py -e 1.0 --dp-clients 1           -> 94.31% (Avg Personalized: 96.41%)  ------------Only machine-1 Active DP
python server.py -e 1.0 --dp-clients 1,2         -> 93.46% (Avg Personalized: 96.41%)  ------------only 1,2 Machine Active DP


************WITHOUT FEDERATED LEARNING (STANDALONE - NO DP)******************

python machine.py --no-dp                        -> 96.89%  ----------------- machine-1                            
                                                 -> 96.82%  ------------------machine-2
                                                 -> 95.51%  ------------------machine-3

python machine1.py --no-dp                       -> 96.89% ------------Machine-1 Without dp      


weight ----------------------------------------------33%


*********************STANDALONE WITH DP (DEFAULT EPSILON = 1.0)***********************

python machine.py                                -> 96.60%  ----------------- machine-1 (DP 1.0)                            
                                                 -> 96.60%  ------------------machine-2 (DP 1.0)
                                                 -> 95.51%  ------------------machine-3 (DP 1.0)

python machine1.py                               -> 96.60% ------------Machine-1 With DP 1.0


*********************COMPARISON EXPERIMENTS (EPSILON = 0.5)***********************
python server.py -e 0.5                          -> 88.49% (Avg Personalized: 96.45%)  ------------All machine DP 0.5
python server.py -e 0.5 --dp-clients 1           -> 93.46% (Avg Personalized: 96.41%)  ------------Only machine-1 Active DP
python server.py -e 0.5 --dp-clients 1,2         -> 90.96% (Avg Personalized: 96.45%)  ------------only 1,2 Machine Active DP
python machine.py -e 0.5                         -> 96.24%  ----------------- machine-1 (DP 0.5)                           
                                                 -> 95.88%  ------------------machine-2 (DP 0.5)
                                                 -> 95.51%  ------------------machine-3 (DP 0.5)
```

---

## 📐 Mathematical Formulation & DP Rigor

### 1. Sensitivity Bounds ($\Delta W$)
Feature vectors $x_i$ are bounded via $L_2$ norm clipping:
$$R = \min\left(\max_i \|x_i\|_2, R_{\text{clip}}\right), \quad R_{\text{clip}} = 5.0$$

The sensitivity of the model weights $W$ under $L_2$ regularized SGD ($\alpha = 0.1$) with local dataset size $N$ is:
$$\Delta W = \frac{2 \cdot R}{N \cdot \alpha}$$

### 2. Sequential Privacy Composition
For $K = 5$ rounds, the per-round privacy budget $\epsilon_r$ is split as:
$$\epsilon_r = \frac{\epsilon_{\text{total}}}{K} = \frac{1.0}{5} = 0.20$$
By the **Sequential Composition Theorem**, total privacy guarantee is:
$$\epsilon_{\text{total}} = \sum_{r=1}^K \epsilon_r = 5 \times 0.20 = 1.00$$

### 3. Laplace Noise Injection
Noise scale $b = \frac{\Delta W}{\epsilon_r}$ calibrates the zero-mean Laplace distribution:
$$W_{\text{private}} = W_{\text{local}} + \text{Laplace}\left(0, \frac{\Delta W}{\epsilon_r}\right)$$

---

## 🛠️ File Structure

| File | Description |
|:---|:---|
| `split_data.py` | Loads `Federeted-data.xlsx` (`Oversampled_Dataset` sheet), applies Dirichlet non-IID splitting ($\alpha=1.0$), and saves client partitions (`machine1_data.json`, `machine2_data.json`, `machine3_data.json`). |
| `server.py` | Central FL Aggregator implementing FedAvg, global weight evaluation, personalization fine-tuning, clinical risk alerts, and visualization generation. |
| `machine1.py`, `machine2.py`, `machine3.py` | Individual client classes handling local model training, norm clipping, and DP noise generation. |
| `machine.py` | Standalone runner script to execute and benchmark all local clients. |
| `run.txt` | Ground-truth text file containing verified experiment accuracy benchmarks. |
| `plot_accuracy.png` | Visualization of FL global accuracy vs round progression. |
| `plot_dp_laplace.png` | Visual depiction of Laplace noise scale vs privacy budget ($\epsilon$). |
| `plot_dataset_sizes.png` | Bar plot of client partition training sizes and class distributions. |
| `plot_confusion_matrix.png` | Combined multi-class confusion matrix (Normal, Mild, Moderate, Severe). |

---

## 🚀 How to Run & Reproduce

### 1. Partition Data (Prerequisite)
```bash
python split_data.py
```

### 2. Run Federated Learning with Default DP ($\epsilon = 1.0$)
```bash
python server.py
```

### 3. Run Federated Learning Without Privacy (Baseline)
```bash
python server.py --no-dp
```

### 4. Run Federated Learning with Custom Privacy Budget ($\epsilon = 0.5$)
```bash
python server.py -e 0.5
```

### 5. Run Partial DP (Only Client 1 Active)
```bash
python server.py -e 1.0 --dp-clients 1
```

### 6. Run Standalone Local Models
```bash
python machine.py --no-dp    # Non-private standalone baseline
python machine.py            # Standalone with default DP (epsilon = 1.0)
```

---

## 🏥 Clinical Health Event Classification & Risk Mapping

The model classifies IoT patient telemetry into 4 health conditions:
1. **Class 0 (Normal):** **LOW RISK** — Routine healthy monitoring.
2. **Class 1 (Mild Event):** **MEDIUM RISK** — Mild vital variation; rest and close monitoring advised.
3. **Class 2 (Moderate Event):** **MEDIUM RISK** — Moderate vital deviation; monitor stress and reduce activity.
4. **Class 3 (Severe Event):** **HIGH RISK** — Critical anomaly detected; immediate clinical consultation alerted.
