# ML Project

This repository contains code and data for multiple experiments and models used in the ML project for the ML-CUP25 dataset (linear, KNN, MLP, SVM, XGBoost, Autoencoder with NN, ensemble methods models).

**Task: Regression with 12 features and 4 targets.**

**Top-level structure**
- `data`: raw datasets and exploration scripts.
- `Monks`: primary implementations for KNN, MLP, SVM, XGBoost, Lightning_Linear for Monks problem.
- `Cup`: primary implementations for KNN, MLP, AE+NN, SVM, XGBoost, Lightning_Linear and Ensemble for Cup.

**Main libraries**
- NumPy: https://numpy.org
- pandas: https://pandas.pydata.org
- Optuna: https://optuna.org/
- scikit-learn: https://scikit-learn.org
- XGBoost: https://xgboost.ai
- PyTorch: https://pytorch.org
- PyTorch Lightning: https://pytorch-lightning.readthedocs.io
- matplotlib: https://matplotlib.org
- seaborn: https://seaborn.pydata.org

**Quick install (recommended: use a virtual environment)**
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install --upgrade pip
pip install -r requirements.txt

# Linux / macOS
python3 -m venv .venv
source .venv/bin/activate
pip install --upgrade pip
pip install -r requirements.txt
```