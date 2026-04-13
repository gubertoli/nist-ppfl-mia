import os
import json
import random
import warnings
import csv
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
from art.estimators.classification import PyTorchClassifier
from xgboost import XGBClassifier
from utils.utils import load_model, load_data, load_hyperparameters, set_seed, extract_meta_features

warnings.filterwarnings("ignore", category=UserWarning)

privacy_tiers = [("cnn", "cnn"), ("dpcnn", "dpcnn10"), ("dpcnn", "dpcnn200")]
os.makedirs("output", exist_ok=True)

for model_type, privacy_type in privacy_tiers:
    set_seed(42) 
    
    print(f"\n{'='*50}\nStarting Stacked Generalization Attack on Tier: {privacy_type}\n{'='*50}")
    
    client_id = "4"
    model_dir = Path(f'problem1/attack_targets/{privacy_type}/client_{client_id}')
    
    with open(Path(model_dir, f'{model_type}_{client_id}_challenge_members.json'), 'r') as f:
        chal_members_4 = json.load(f)['challenge members']
    with open(Path(model_dir, f'{model_type}_{client_id}_relevant_members.json'), 'r') as f:
        rel_members_4 = json.load(f)['relevant members']
        
    rel_x, rel_y = load_data(Path(model_dir, f'{model_type}_{client_id}_relevant_records.dat'))
    ext_x, ext_y = load_data(Path(model_dir, f'{model_type}_{client_id}_external_records.dat'))
    cha_x, cha_y = load_data(Path(model_dir.parent, f'{privacy_type}_challenge_records.dat'))
    
    task_model_4 = load_model(Path(model_dir, f'{model_type}_{client_id}.torch'), rel_x.shape[1], privacy_type)
    hyperparams_4 = load_hyperparameters(Path(model_dir, f'{model_type}_{client_id}_hyperparameters.json'))
    
    print(f"[{privacy_type}] Extracting meta-features for Client 4 (Ground Truth)...")
    client4_data = extract_meta_features(rel_x, rel_y, ext_x, ext_y, cha_x, cha_y, task_model_4, hyperparams_4, client_id, privacy_type, rel_members_4, chal_members_4)
    
    # Train the Base Meta-Model on Client 4
    mask_train = client4_data["label"] != -1
    feature_cols = ["loss", "nn_proba", "rf_proba", "dt_proba", "gb_proba", "knn_proba", "svm_proba", "lr_proba"]
    X_train = np.column_stack([client4_data[col][mask_train] for col in feature_cols])
    y_train = client4_data["label"][mask_train]
    
    meta_model = XGBClassifier(eval_metric='logloss', random_state=42)
    meta_model.fit(X_train, y_train)
    
    challenge_predictions = {}
    challenge_predictions["pred4_prob"] = np.zeros(len(cha_x))
    challenge_predictions["pred4_prob"][chal_members_4] = 1.0
    
    dynamic_thresholds = {}
    
    for client_id in ["1", "2", "3"]:
        print(f"[{privacy_type}] Extracting meta-features and inferring Client {client_id}...")
        model_dir = Path(f'problem1/attack_targets/{privacy_type}/client_{client_id}')
        
        rel_x, rel_y = load_data(Path(model_dir, f'{model_type}_{client_id}_relevant_records.dat'))
        ext_x, ext_y = load_data(Path(model_dir, f'{model_type}_{client_id}_external_records.dat'))
        
        task_model = load_model(Path(model_dir, f'{model_type}_{client_id}.torch'), rel_x.shape[1], privacy_type)
        hyperparams = load_hyperparameters(Path(model_dir, f'{model_type}_{client_id}_hyperparameters.json'))
        
        client_data = extract_meta_features(rel_x, rel_y, ext_x, ext_y, cha_x, cha_y, task_model, hyperparams, client_id, privacy_type)
        
        mask_ext = client_data["type"] == "ext"
        X_ext = np.column_stack([client_data[col][mask_ext] for col in feature_cols])
        y_ext = client_data["label"][mask_ext]
        
        adapted_model = XGBClassifier(eval_metric='logloss', random_state=42)
        adapted_model.fit(X_ext, y_ext, xgb_model=meta_model)
        
        mask_challenge = client_data["type"] == "cha"
        X_challenge = np.column_stack([client_data[col][mask_challenge] for col in feature_cols])
        challenge_predictions[f"pred{client_id}_prob"] = adapted_model.predict_proba(X_challenge)[:, 1]

    for c in ["1", "2", "3"]:
        dynamic_thresholds[c] = np.percentile(
             challenge_predictions[f"pred{c}_prob"], 55
        )
        print(f"    -> Column threshold for Client {c}: {dynamic_thresholds[c]:.6f}")

    print(f"[{privacy_type}] Resolving predictions and creating submission file...")
    final_preds = []
    
    for idx in range(len(cha_x)):
        if challenge_predictions["pred4_prob"][idx] == 1.0:
            final_preds.append(4)
            continue
            
        row_mean = np.mean([challenge_predictions[f"pred{c}_prob"][idx]
                        for c in ["1", "2", "3"]])
        valid_clients = {}
        for c in ["1", "2", "3"]:
            prob = challenge_predictions[f"pred{c}_prob"][idx]
            if prob > dynamic_thresholds[c] and prob > 1.5*row_mean:
                valid_clients[int(c)] = prob

        if not valid_clients:
            final_preds.append(0)
        else:
            best_client = max(valid_clients, key=valid_clients.get)
            final_preds.append(best_client)

    template_path = f"problem1/attack_targets/{privacy_type}/{privacy_type}_submission_file.csv"
    with open(template_path, 'r') as f:
        reader = csv.reader(f)
        header = next(reader)
        
    out_file = f"output/{privacy_type}_ensemble_detailed_submission.csv"
    with open(out_file, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['index', 'prediction', 'pred1', 'pred2', 'pred3', 'pred4'])
        for i in range(len(cha_x)):
            writer.writerow([
                i, 
                final_preds[i],
                challenge_predictions["pred1_prob"][i],
                challenge_predictions["pred2_prob"][i],
                challenge_predictions["pred3_prob"][i],
                challenge_predictions["pred4_prob"][i]
            ])
            
    print(f"✔ Saved detailed submission for {privacy_type} to: {out_file}")
