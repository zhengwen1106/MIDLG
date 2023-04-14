MIDLG: Mutual Information based Dual Level GNN for Transaction Fraud Complaint Verification
====
MIDLG is a graph model designed to solve the 'transaction fraud" complaint verification problem, which verifys whether a transaction corresponding to a complaint is fraudulent to prevent economic 

To address "Transaction fraud" complaint verification problem, MIDLG defines a complaint as a super-node consisting of involved individuals, and characterizes the individual over node-level and super-node-level. Furthermore, the mutual information minimization objective is proposed based on “complaint verification-causal graph” to decouple the model prediction from relying on specific fraud ways, and thus achieve stability.

Due to the company's data security and user privacy, the dataset couldn't be provided. We provide randomized data to ensure the model runs.


## Requirements

  * PyTorch 1.11
  * Python 3.8
  * scikit-learn 1.1.1 

## Usage

```python train.py```
