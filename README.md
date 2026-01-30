# Graph-Based Fraud Detection using Graph Neural Networks (GNN)

This project implements an end-to-end **graph analytics pipeline for fraud / money laundering detection** using **Graph Neural Networks (GNNs)**.

The entire project is implemented in a **Jupyter Notebook (VS Code)** and demonstrates how graph structure and node features can be leveraged to detect fraudulent behavior in highly imbalanced datasets.

---

## 🚀 Features
- Graph construction using NetworkX
- Graph Neural Network (GCN) using PyTorch Geometric
- Class imbalance handling:
  - Weighted loss
  - Focal loss
  - Threshold tuning
- Fraud-focused evaluation (Precision, Recall, ROC-AUC)
- Automated visualization generation
- Automatic PDF report generation

---

## 📊 Final Model Performance
- Fraud Recall: **71.43%**
- Fraud Precision: **83.58%**
- ROC-AUC: **0.83**
- Threshold-tuned for optimal fraud detection

> Note: Accuracy is not the primary metric due to class imbalance.

---

## 🧠 Techniques Used
- Graph Convolutional Networks (GCN)
- Weighted Cross-Entropy Loss
- Focal Loss
- Threshold Optimization
- SMOTE (used only during experimentation)

---

## 📁 Project Structure
fraud_grapg_project/
│── fraud_gnn.ipynb
│── Fraud_Analytics_Report.pdf
│── training_history_FIXED.png
│── confusion_matrix_FIXED.png
│── precision_recall_curve.png
│── graph_visualization.png
│── requirements.txt
│── README.md


---

## ⚙️ Setup Instructions

```bash
python -m venv venv
source venv/bin/activate      # Linux / Mac
venv\Scripts\activate         # Windows

pip install -r requirements.txt


Run the notebook:

jupyter notebook fraud_gnn.ipynb

📌 Notes

SMOTE was explored during experimentation.

Final GNN performance relies on imbalance-aware loss functions to preserve graph structure.

The project can be extended to real-world AML datasets (Elliptic, bank transactions).