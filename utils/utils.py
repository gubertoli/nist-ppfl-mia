import random
import pickle as pkl
import json
import torch
import torch.nn as nn
import torch.optim as optim
from pathlib import Path
import numpy as np
from art.attacks.inference.membership_inference import MembershipInferenceBlackBox
from art.estimators.classification import PyTorchClassifier


def set_seed(seed=42):
    """Locks all random seeds for 100% reproducible ART features."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def extract_meta_features(rel_x, rel_y, ext_x, ext_y, challenge_x, challenge_y, task_model, hyperparams, client_id, privacy_type, relevant_members=None, challenge_members=None):
    """Shared feature extraction pipeline for both attack and evaluation scripts."""
    loss_fn = nn.CrossEntropyLoss()
    
    def process_records(x_data, y_data, true_members, record_type, is_challenge=False, is_external=False):
        records = []
        for i, (x, y) in enumerate(zip(x_data, y_data)):
            x_tensor = torch.tensor(x).unsqueeze(0).unsqueeze(0)
            with torch.no_grad():
                prediction = task_model(x_tensor)
            calc_loss = loss_fn(prediction, torch.tensor([y], dtype=torch.long)).item()
            
            if is_external:
                label = 0
            elif true_members is not None:
                label = 1 if i in true_members else 0
            else:
                label = -1 
                
            records.append({"loss": calc_loss, "label": label, "type": record_type})
        return records

    records_rel = process_records(rel_x, rel_y, relevant_members, "rel")
    records_ext = process_records(ext_x, ext_y, None, "ext", is_external=True)
    records_cha = process_records(challenge_x, challenge_y, challenge_members, "cha", is_challenge=True)
    
    losses = [r['loss'] for r in records_rel + records_ext + records_cha]
    labels = [r['label'] for r in records_rel + records_ext + records_cha]
    types = [r['type'] for r in records_rel + records_ext + records_cha]
    
    dataset = {
        "loss": np.array(losses, dtype=np.float32),
        "label": np.array(labels, dtype=np.int32),
        "type": np.array(types)
    }
    
    criterion = nn.CrossEntropyLoss()
    optimizer_name = hyperparams['optimizer']
    learning_rate = hyperparams['learning rate']
    weight_decay = hyperparams['weight decay']
    num_classes = hyperparams['total classes']

    if optimizer_name == 'sgd':
        optimizer = optim.SGD(task_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
    else:
        optimizer = optim.Adamax(task_model.parameters(), lr=learning_rate, weight_decay=weight_decay)
        
    classifier = PyTorchClassifier(model=task_model, loss=criterion, optimizer=optimizer, input_shape=(rel_x.shape[1],), nb_classes=num_classes)

    attack_models = {
        "nn": MembershipInferenceBlackBox(classifier, attack_model_type="nn", nn_model_epochs=100, nn_model_batch_size=100, nn_model_learning_rate=0.03),
        "rf": MembershipInferenceBlackBox(classifier, attack_model_type="rf"),
        "dt": MembershipInferenceBlackBox(classifier, attack_model_type="dt"),
        "gb": MembershipInferenceBlackBox(classifier, attack_model_type="gb"),
        "knn": MembershipInferenceBlackBox(classifier, attack_model_type="knn"),
        "svm": MembershipInferenceBlackBox(classifier, attack_model_type="svm"),
        "lr": MembershipInferenceBlackBox(classifier, attack_model_type="lr")
    }

    rel_x_preds = task_model(torch.tensor(rel_x).unsqueeze(1)).squeeze().detach().numpy()
    ext_x_preds = task_model(torch.tensor(ext_x).unsqueeze(1)).squeeze().detach().numpy()

    for name, attack_model in attack_models.items():
        attack_model.fit(x=rel_x, y=rel_y, test_x=ext_x, test_y=ext_y, pred=rel_x_preds, test_pred=ext_x_preds)
        
        rel_proba = attack_model.infer(x=rel_x, y=rel_y, probabilities=True)
        ext_proba = attack_model.infer(x=ext_x, y=ext_y, probabilities=True)
        cha_proba = attack_model.infer(x=challenge_x, y=challenge_y, probabilities=True)
        
        dataset[f"{name}_proba"] = np.concatenate([rel_proba, ext_proba, cha_proba])

    return dataset


def load_model(model_path: Path, num_data_features: int, privacy_tier: str):
    """Loads the specified PyTorch model architecture and weights dynamically."""
    
    if privacy_tier == 'cnn':
        from problem1.attack_targets.cnn.model import Net
    elif privacy_tier == 'dpcnn10':
        from problem1.attack_targets.dpcnn10.model import Net
    elif privacy_tier == 'dpcnn200':
        from problem1.attack_targets.dpcnn200.model import Net
    else:
        raise ValueError(f"Unknown privacy_tier: {privacy_tier}")

    model = Net(num_data_features)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model

def load_data(data_path: Path):
    """Loads and preprocesses the genomic feature vectors and labels."""
    with open(data_path, "rb") as f:
        data = pkl.load(f)
    features = len(data[0]) - 1
    x = data[:, :features].astype(np.float32)
    y = data[:, features].astype(np.int64)
    return x, y

def load_hyperparameters(hyperparameters_path: Path):
    """Loads the JSON hyperparameters file."""
    with open(hyperparameters_path, 'r') as f:
        hyperparameters = json.load(f)
    return hyperparameters
