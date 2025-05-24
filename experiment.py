from dataloader import *
from unimodal_baseline import *

import torch
import torch.nn as nn
import torch.optim as optim

import numpy as np
from copy import deepcopy
from itertools import product

from sklearn.exceptions import NotFittedError
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split
from sklearn.utils.validation import check_is_fitted


def is_fitted(model):
    try:
        check_is_fitted(model)
        return True
    except NotFittedError:
        return False

class EarlyFusion:
    '''early fusion where we extract image embeddings and concatenate 
    with tabular feature vectors to train a fully connected classifier on com-
    bined representation
    

    1. Use or train unimodal models 
    2. Extract features for all samples in train and test
    3. Get RF probabilities for tabular features
    4. Concatenate features
    5. Train MLP on fused features
    '''
    def __init__(self, device=None, tab_model=None, mri_model=None,
                 mlp_hidden_dims=(64, 32), dropout=0.3, lr=5e-3, epochs=20, patience=5):
        self.tab_model = tab_model if tab_model is not None else TabularBaseline()
        self.mri_model = mri_model if mri_model is not None else MRIBaseline()
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.fusion_mlp = None

        self.mlp_hidden_dims = mlp_hidden_dims
        self.dropout = dropout
        self.lr = lr
        self.epochs = epochs
        self.patience = patience

    def fit(self, train_loader, val_loader, test_loader, slice=6):
        # Train tabular model if needed
        if not is_fitted(self.tab_model.model):
            print("Training tabular (RF) baseline...")
            self.tab_model.load_data(train_loader, test_loader)
            self.tab_model.train()
        else:
            print("Using pre-trained tabular (RF) baseline...")

        # Train MRI model if needed
        if self.mri_model.model is None:
            print("Training MRI (CNN) baseline...")
            self.mri_model.load_data(train_loader, test_loader)
            if slice == -1:
                self.mri_model.train_multi_slice_model(train_loader, test_loader, num_slices=9)
                self.mri_model = copy.deepcopy(self.mri_model.multi_slice_model)
            else:
                self.mri_model.train_on_specific_slice(train_loader, val_loader, test_loader, slice_idx=slice)
        else:
            print("Using pre-trained MRI (CNN) baseline...")

        self.mri_model.model.eval()

        # Extract features
        print("Extracting features for fusion...")
        X_train_cnn, X_train_tab, y_train = self._extract_features(train_loader, self.mri_model, slice_idx=slice)
        X_val_cnn, X_val_tab, y_val = self._extract_features(val_loader, self.mri_model, slice_idx=slice)
        X_test_cnn, X_test_tab, y_test = self._extract_features(test_loader, self.mri_model, slice_idx=slice)

        # Get RF probabilities for tabular data
        X_train_rf = self.tab_model.model.predict_proba(X_train_tab)
        X_val_rf = self.tab_model.model.predict_proba(X_val_tab)
        X_test_rf = self.tab_model.model.predict_proba(X_test_tab)

        # Concatenate
        X_train_fused = np.concatenate([X_train_cnn, X_train_rf], axis=1)
        X_val_fused = np.concatenate([X_val_cnn, X_val_rf], axis=1)
        X_test_fused = np.concatenate([X_test_cnn, X_test_rf], axis=1)

        # Train MLP
        print("Training fusion MLP...")
        self.fusion_mlp = self._build_mlp(X_train_fused.shape[1], 2).to(self.device)
        self._train_mlp(self.fusion_mlp, X_train_fused, y_train, X_val_fused, y_val)

    def _extract_features(self, loader, cnn_model, slice_idx=6):
        cnn_features = []
        tabular_features = []
        labels = []

        cnn_model.model.eval()

        for batch in loader:
            slice_img = batch['mri'][:, :, slice_idx, :, :].to(self.device)
            with torch.no_grad():
                x = cnn_model.model.features(slice_img)
                x = cnn_model.model.avgpool(x)
                x = torch.flatten(x, 1)
                x = cnn_model.model.classifier[0](x)
                x = cnn_model.model.classifier[1](x)
                x = cnn_model.model.classifier[2](x)
                x = cnn_model.model.classifier[3](x)
                x = cnn_model.model.classifier[4](x)
                feats = x.cpu().numpy()

            cnn_features.append(feats)
            tabular_features.append(batch['tabular'].cpu().numpy())
            labels.append(batch['label'].view(-1).cpu().numpy())

        X_cnn = np.concatenate(cnn_features, axis=0)
        X_tab = np.concatenate(tabular_features, axis=0)
        y = np.concatenate(labels, axis=0)
        return X_cnn, X_tab, y

    def _build_mlp(self, input_dim, num_classes):
        layers = []
        dims = [input_dim] + list(self.mlp_hidden_dims)
        for i in range(len(dims) - 1):
            layers.append(nn.Linear(dims[i], dims[i + 1]))
            layers.append(nn.ReLU())
            layers.append(nn.Dropout(self.dropout))
        layers.append(nn.Linear(dims[-1], num_classes))
        return nn.Sequential(*layers)

    def _train_mlp(self, model, X_train, y_train, X_val, y_val):
        model.train()
        optimizer = optim.Adam(model.parameters(), lr=self.lr)
        criterion = nn.CrossEntropyLoss()
        X_train = torch.FloatTensor(X_train).to(self.device)
        y_train = torch.LongTensor(y_train).to(self.device)
        X_val = torch.FloatTensor(X_val).to(self.device)
        y_val = torch.LongTensor(y_val).to(self.device)
        best_acc = 0
        patience_counter = 0

        for epoch in range(self.epochs):
            optimizer.zero_grad()
            logits = model(X_train)
            loss = criterion(logits, y_train)
            loss.backward()
            optimizer.step()

            # Evaluate on val
            model.eval()
            with torch.no_grad():
                val_logits = model(X_val)
                val_preds = torch.argmax(val_logits, dim=1)
                acc = (val_preds == y_val).float().mean().item()
            print(f"Epoch {epoch+1}: Loss={loss.item():.4f}, Val Acc={acc:.3f}")

            if acc > best_acc:
                best_acc = acc
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print("Early stopping for fusion MLP.")
                    break
            model.train()

    def evaluate(self, test_loader, slice_idx=6):
        if self.fusion_mlp is None:
            raise ValueError("Fusion MLP not trained. Call fit() first.")
        cnn_model = self.mri_model.model
        cnn_model.eval()
        X_test_cnn, X_test_tab, y_test = self._extract_features(test_loader, self.mri_model, slice_idx)
        X_test_rf = self.tab_model.model.predict_proba(X_test_tab)
        X_test_fused = np.concatenate([X_test_cnn, X_test_rf], axis=1)

        self.fusion_mlp.eval()
        X_tensor = torch.FloatTensor(X_test_fused).to(self.device)
        with torch.no_grad():
            logits = self.fusion_mlp(X_tensor)
            preds = torch.argmax(logits, dim=1).cpu().numpy()

        acc = accuracy_score(y_test, preds)
        f1 = f1_score(y_test, preds)
        prec = precision_score(y_test, preds)
        rec = recall_score(y_test, preds)
        print(f"Fusion Model - Accuracy: {acc:.3f}, F1: {f1:.3f}, Precision: {prec:.3f}, Recall: {rec:.3f}")
        return {'accuracy': acc, 'f1': f1, 'precision': prec, 'recall': rec}
    
    
    def grid_search_early_fusion(self, param_grid = {'mlp_hidden_dims': [(64, ), (64,32), (128, 64), (256, 128)],
                                'dropout': [0.3, 0.5],'lr': [1e-3, 5e-4],'epochs': [20],'patience': [5]},  slice_idx=6):
            all_results = []
            top_models = []

            keys, values = zip(*param_grid.items())
            combinations = list(product(*values))

            for combo in combinations:
                params = dict(zip(keys, combo))
                print(f"\n>> Training with params: {params}")

                model = EarlyFusion(
                    tab_model = self.tab_model,
                    mri_model = self.mri_model,
                    mlp_hidden_dims=params['mlp_hidden_dims'],
                    dropout=params['dropout'],
                    lr=params['lr'],
                    epochs=params['epochs'],
                    patience=params['patience']
                )

                model.fit(self.train_loader, self.val_loader, self.test_loader, slice=slice_idx)
                metrics = model.evaluate(self.val_loader, slice_idx=slice_idx)

                all_results.append((params, metrics, deepcopy(model)))

            all_results.sort(key=lambda x: x[1]['f1'], reverse=True)
            top_models = all_results[:3]
            return top_models, all_results
    
class LateFusion:
    '''late fusion where we train a CNN and tabular model independently, 
    combining class probability outputs using simple averaging and/or 
    weighted averaging'''
    def __init__(self, mri_model, tab_model, device=None):
        self.mri_model = mri_model
        self.tab_model = tab_model
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.best_weight = 0.5
        self.fusion_models = {}
        self.best_mlp_config = None

    def extract_features(self, loader, slice_idx=6):
        cnn_probs, rf_probs, labels = [], [], []
        self.mri_model.eval()
        with torch.no_grad():
            for batch in loader:
                imgs = batch['mri'][:, :, slice_idx, :, :].to(self.device)
                outputs = self.mri_model(imgs)
                probs = torch.softmax(outputs, dim=1).cpu().numpy()
                cnn_probs.append(probs)
                
                tab_feats = batch['tabular'].cpu().numpy()
                rf_probs.append(self.tab_model.model.predict_proba(tab_feats))
                labels.append(batch['label'].view(-1).cpu().numpy())
        return np.concatenate(cnn_probs), np.concatenate(rf_probs), np.concatenate(labels)

    def tune_weighted_average(self, cnn_probs_val, rf_probs_val, val_labels):
        best_f1, best_weight = -1, 0.5
        for w in np.linspace(0, 1, 21):
            blended = w * rf_probs_val + (1 - w) * cnn_probs_val
            preds = np.argmax(blended, axis=1)
            f1 = f1_score(val_labels, preds)
            if f1 > best_f1:
                best_f1, best_weight = f1, w
        self.best_weight = best_weight

    def tune_mlp(self, cnn_train, rf_train, train_labels, cnn_val, rf_val, val_labels):
        param_grid = {
            'hidden_layer_sizes': [(64,), (128,), (128, 64), (256, 128)],
            'learning_rate_init': [0.001, 0.0005]
        }
        results = []
        for hls in param_grid['hidden_layer_sizes']:
            for lr in param_grid['learning_rate_init']:
                mlp = MLPClassifier(hidden_layer_sizes=hls, learning_rate_init=lr, max_iter=500, random_state=42)
                mlp.fit(np.concatenate([cnn_train, rf_train], axis=1), train_labels)
                preds = mlp.predict(np.concatenate([cnn_val, rf_val], axis=1))
                f1 = f1_score(val_labels, preds)
                results.append({'hidden_layers': hls, 'lr': lr, 'f1': f1, 'model': mlp})
        best = max(results, key=lambda x: x['f1'])
        self.best_mlp_config = best

    def train_all_models(self, cnn_train, rf_train, train_labels):
        X_train = np.concatenate([cnn_train, rf_train], axis=1)
        self.fusion_models = {
            "Logistic Regression": LogisticRegression(max_iter=1000).fit(X_train, train_labels),
            "Random Forest": RandomForestClassifier(n_estimators=100, max_depth=5, max_features='sqrt').fit(X_train, train_labels),
            "MLP": self.best_mlp_config['model'],
            "Simple Average": None,
            "Weighted Average": None
        }

    def predict_all(self, cnn_probs_test, rf_probs_test):
        X_test = np.concatenate([cnn_probs_test, rf_probs_test], axis=1)
        predictions = {}
        for name, model in self.fusion_models.items():
            if model is not None:
                preds = model.predict(X_test)
            else:
                avg_probs = (cnn_probs_test + rf_probs_test) / 2
                if name == "Weighted Average":
                    avg_probs = self.best_weight * rf_probs_test + (1 - self.best_weight) * cnn_probs_test
                preds = np.argmax(avg_probs, axis=1)
            predictions[name] = preds
        return predictions

    def evaluate(self, predictions, true_labels):
        metrics = {}
        for name, preds in predictions.items():
            metrics[name] = {
                "Accuracy": accuracy_score(true_labels, preds),
                "Precision": precision_score(true_labels, preds),
                "Recall": recall_score(true_labels, preds),
                "F1 Score": f1_score(true_labels, preds)
            }
        return metrics

    def per_cdr_accuracy(self, predictions, true_labels, cdr_labels):
        cdr_values = np.unique(cdr_labels)
        per_model = {}
        for name, preds in predictions.items():
            accs = []
            for cdr in cdr_values:
                mask = np.array(cdr_labels) == cdr
                if np.sum(mask) == 0:
                    accs.append(np.nan)
                else:
                    accs.append(accuracy_score(np.array(true_labels)[mask], np.array(preds)[mask]))
            per_model[name] = accs
        return cdr_values, per_model
