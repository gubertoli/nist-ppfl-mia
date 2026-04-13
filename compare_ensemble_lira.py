import os
import json
import warnings
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
from art.estimators.classification import PyTorchClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_predict, StratifiedKFold
from sklearn.metrics import roc_curve
from utils.utils import load_model, load_data, load_hyperparameters, set_seed, extract_meta_features
os.environ['PYTHONHASHSEED'] = '42'
os.environ['OMP_NUM_THREADS'] = '1'
os.environ['MKL_NUM_THREADS'] = '1'

warnings.filterwarnings("ignore")

privacy_tiers = [("cnn", "cnn", "#d62728"), ("dpcnn", "dpcnn10", "#1f77b4"), ("dpcnn", "dpcnn200", "#ff7f0e")]

print(f"{'='*50}\nComparative TPR @ Low FPR Evaluation\n{'='*50}")

for model_type, privacy_type, color in privacy_tiers:
    set_seed(42)
    client_id = "4"
    model_dir = Path(f'problem1/attack_targets/{privacy_type}/client_{client_id}')
    
    with open(Path(model_dir, f'{model_type}_{client_id}_relevant_members.json'), 'r') as f:
        rel_members_4 = json.load(f)['relevant members']
    with open(Path(model_dir, f'{model_type}_{client_id}_challenge_members.json'), 'r') as f:
        chal_members_4 = json.load(f)['challenge members']
        
    rel_x, rel_y = load_data(Path(model_dir, f'{model_type}_{client_id}_relevant_records.dat'))
    ext_x, ext_y = load_data(Path(model_dir, f'{model_type}_{client_id}_external_records.dat'))
    challenge_data_path = Path(model_dir.parent, f'{privacy_type}_challenge_records.dat')
    cha_x, cha_y = load_data(challenge_data_path)
    
    task_model_4 = load_model(Path(model_dir, f'{model_type}_{client_id}.torch'), rel_x.shape[1], privacy_type)
    hyperparams_4 = load_hyperparameters(Path(model_dir, f'{model_type}_{client_id}_hyperparameters.json'))
    
    print(f"\nProcessing {privacy_type}...")
    dataset = extract_meta_features(rel_x, rel_y, ext_x, ext_y, cha_x, cha_y, task_model_4, hyperparams_4, client_id, privacy_type, rel_members_4, chal_members_4)
    
    num_members = (dataset["label"] == 1).sum()
    num_non_members = (dataset["label"] == 0).sum()
    print(f"  Dataset Stats: {len(dataset['label'])} total samples | {num_members} IN (members) | {num_non_members} OUT (non-members)")
    
    # ----------------------------------------------------
    # 1. Evaluate Ensemble
    # ----------------------------------------------------
    feature_cols = ["loss", "nn_proba", "rf_proba", "dt_proba", "gb_proba", "knn_proba", "svm_proba", "lr_proba"]
    X = np.column_stack([dataset[col] for col in feature_cols])
    y = dataset["label"]
    
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
    meta_model = XGBClassifier(eval_metric='logloss', random_state=42)
    y_scores_ensemble = cross_val_predict(meta_model, X, y, cv=cv, method='predict_proba')[:, 1]
    fpr_ens, tpr_ens, _ = roc_curve(y, y_scores_ensemble)
    
    # ----------------------------------------------------
    # 2. Evaluate LiRA baseline
    # ----------------------------------------------------
    ext_mask = dataset["type"] == "ext"
    losses_ext = dataset["loss"][ext_mask]
    
    mean_loss_out = np.mean(losses_ext)
    std_loss_out = np.std(losses_ext)
    
    y_scores_lira = (mean_loss_out - dataset["loss"]) / (std_loss_out + 1e-9)
    fpr_lira, tpr_lira, _ = roc_curve(y, y_scores_lira)
    
    def get_tpr_at_fpr(fpr_array, tpr_array, target_fpr):
        """Returns the highest TPR achievable strictly at or below the target FPR."""
        valid_indices = np.where(fpr_array <= target_fpr)[0]
        if len(valid_indices) == 0:
            return 0.0
        return tpr_array[valid_indices[-1]]

    target_1_percent = 0.0101 # Slightly above 0.01 to safely capture the 1/103 (0.97%) discrete step
    target_3_percent = 0.0301 # Slightly above 0.03 to safely capture the 3/103 (2.91%) discrete step

    tpr_ens_1 = get_tpr_at_fpr(fpr_ens, tpr_ens, target_1_percent)
    tpr_lira_1 = get_tpr_at_fpr(fpr_lira, tpr_lira, target_1_percent)
    
    tpr_ens_3 = get_tpr_at_fpr(fpr_ens, tpr_ens, target_3_percent)
    tpr_lira_3 = get_tpr_at_fpr(fpr_lira, tpr_lira, target_3_percent)

    print(f"  [Ensemble] TPR @ <=1.00% FPR: {tpr_ens_1*100:>5.2f}%  |  TPR @ <=3.00% FPR: {tpr_ens_3*100:>5.2f}%")
    print(f"  [LiRA]     TPR @ <=1.00% FPR: {tpr_lira_1*100:>5.2f}%  |  TPR @ <=3.00% FPR: {tpr_lira_3*100:>5.2f}%")
    
