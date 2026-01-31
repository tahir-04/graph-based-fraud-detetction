# Graph-Based Fraud Detection using Graph Neural Networks (GNN)

This project implements an **end-to-end graph analytics pipeline for fraud and money laundering detection** using **Graph Neural Networks (GNNs)**.  
The system models transactional data as a graph and leverages **graph structure + node features** to identify fraudulent entities in **highly imbalanced datasets**.

The entire project is implemented in a **Jupyter Notebook (VS Code)** and focuses on **real-world fraud detection challenges**, including class imbalance, recall optimization, and decision threshold tuning.

---

## Key Highlights
- End-to-end **Graph-based Fraud Detection System**
- Graph construction using **NetworkX**
- Fraud classification using **Graph Convolutional Networks (GCN)** with **PyTorch Geometric**
- Advanced **class imbalance handling**
- Fraud-focused evaluation metrics
- Automated visualizations and **PDF report generation**
- Fully reproducible and GitHub-ready

---

##  Problem Statement
Traditional fraud detection methods struggle to capture **relational patterns** between entities.  
This project addresses that limitation by modeling data as a **graph**, where:

- **Nodes** → Accounts / Transactions  
- **Edges** → Money transfers or relationships  

Graph Neural Networks exploit both **feature information** and **network structure** to detect coordinated or hidden fraud patterns.

---

##  Project Architecture

1. Synthetic transaction graph generation (scale-free network)
2. Node feature engineering and fraud labeling
3. Graph construction using NetworkX
4. Conversion to PyTorch Geometric data format
5. GNN model training (GCN)
6. Class imbalance mitigation
7. Model evaluation & visualization
8. Automated PDF fraud analytics report

---

##  Techniques Used

### Graph Modeling
- NetworkX graph construction
- Power-law (Barabási–Albert) transaction network

### Machine Learning
- Graph Convolutional Network (GCN)
- Weighted Cross-Entropy Loss
- Focal Loss
- Decision Threshold Tuning

### Imbalance Handling
- Fraud Recall Monitoring
- Threshold Optimization
- SMOTE (used during experimentation only)

### Evaluation Metrics
- Precision, Recall, F1-score (Fraud-focused)
- ROC-AUC
- Confusion Matrix
- Precision–Recall Curve

---

##  Final Model Performance (Test Set)

| Metric | Value |
|------|------|
| Fraud Recall | **71.43%** |
| Fraud Precision | **83.58%** |
| ROC-AUC | **0.83** |
| Overall Accuracy | 75.19% |

> **Note:** Accuracy is not the primary metric due to severe class imbalance.  
> Fraud Recall and ROC-AUC are prioritized.

---

##  Generated Artifacts
The project automatically generates the following outputs:

- `Fraud_Analytics_Report.pdf`
- `training_history_FIXED.png`
- `confusion_matrix_FIXED.png`
- `precision_recall_curve.png`
- Graph visualization of fraud vs legitimate nodes

---

##  Project Structure
graph-based-fraud-detection/
│── fraud_gnn.ipynb
│── Fraud_Analytics_Report.pdf
│── training_history_FIXED.png
│── confusion_matrix_FIXED.png
│── precision_recall_curve.png
│── graph_visualization.png
│── requirements.txt
│── README.md


---

##  Setup Instructions

### 1️ Clone Repository
```bash
git clone https://github.com/tahir-04/graph-based-fraud-detetction.git
cd graph-based-fraud-detetction
2️⃣ Create Virtual Environment
python -m venv venv
source venv/bin/activate      # Linux / Mac
venv\Scripts\activate         # Windows
3️⃣ Install Dependencies
pip install -r requirements.txt
4️⃣ Run Notebook
jupyter notebook fraud_gnn.ipynb
🧪 Notes on SMOTE Usage
SMOTE was explored only during experimentation

Final GNN performance relies on imbalance-aware loss functions

This preserves graph topology integrity, which is critical for GNNs

 Future Enhancements
Integration with real AML datasets (Elliptic Bitcoin, banking transactions)

Graph Attention Networks (GAT)

GNNExplainability for fraud reasoning

Streaming transaction inference

Interactive dashboard (Streamlit / Dash)

 Author
Mohamed Tahir
