from dataloader import *

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import copy
# import shap

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from torchvision import models
from torchvision.models import vgg16, VGG16_Weights

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer, accuracy_score, precision_score, recall_score, f1_score, confusion_matrix

'''
Random forest classifier for tabular baseline model
- Init parameters: num estimators, max depth, max features, and random state
- Methods:
    load_data -- give the dataloaders and it populates the .X_tktk 
                 and .y_tktk attributes
    train -- fits RF to training set
    evaluate -- given test boolean to decide whether to evaluate on 
                test or validation set. returns key metrics in a dictionary and prints stuff
    gridSearch -- tests every combo of hyperparameters on test set to find best ones
                final selection of model is based on performance on validation set though
    _evaluate_specific_model -- helper method for gridSearch comparison of diff models
'''
class TabularBaseline:
    def __init__(self, n_estimators= 100, max_depth = 5, max_features = 'sqrt', random_state = 42):
        self.model = RandomForestClassifier(
                    n_estimators=n_estimators, 
                    max_depth=max_depth, 
                    max_features=max_features, 
                    random_state=random_state
                    )
        self.X_train = None
        self.y_train = None
        self.X_val = None
        self.X_test = None
        self.y_test = None
        self.y_val = None
        self.feature_names = ['Age', 'Sex', 'Education', 'Hand', 'MMSE']

    def load_data(self, train_loader, test_loader, val_loader):
        # Load training data
        X, y = [], []
        for batch in tqdm(train_loader, desc="Loading training data"):
            X.extend(batch['tabular'].numpy())
            y.extend(batch['label'].numpy())
        self.X_train = np.array(X)
        self.y_train = np.array(y).ravel() #in case array isn't 1d
        
        # print("Number of features:", self.X_train.shape[1])
        
        # Load test data
        X, y = [], []
        for batch in tqdm(test_loader, desc="Loading test data"):
            X.extend(batch['tabular'].numpy())
            y.extend(batch['label'].numpy())
        self.X_test = np.array(X)
        self.y_test = np.array(y).ravel()

        # Load validation data
        X, y = [], []
        for batch in tqdm(val_loader, desc="Loading validation data"):
            X.extend(batch['tabular'].numpy())
            y.extend(batch['label'].numpy())
        self.X_val = np.array(X)
        self.y_val = np.array(y).ravel()
        
    def train(self):
        if self.X_train is None or self.y_train is None:
            return 
        
        self.model.fit(self.X_train, self.y_train)
        
        # # SHAP Stuff, commenting out for now because it's slow and annoying
        # explainer = shap.TreeExplainer(self.model)
        # shap_values = explainer.shap_values(self.X_test)  # Use test data for visualization

        # # Generate waterfall plots for 5 randomly seleccted patients
        # sample_indices = np.random.choice(len(self.X_test), min(5, len(self.X_test)), replace=False)
        
        # for i, idx in enumerate(sample_indices):
        #     plt.figure(figsize=(10, 6))

        #     shap.plots.force(explainer.expected_value[1], 
        #                    shap_values[idx, :, 1].flatten(), 
        #                    self.X_test[idx], 
        #                    feature_names=self.feature_names,
        #                    matplotlib=True)
        #     plt.title(f'Patient {i+1} SHAP Values')
        #     plt.tight_layout()
        #     #fig(f'tabular_shap_waterfall_patient_{i+plt.save1}.png')
        #     plt.close()

    def evaluate(self, test = False):
        """Evaluate the model on test data and print detailed metrics."""
        if self.X_test is None or self.y_test is None:
            return

        # Get predictions
        if test:
            y_pred = self.model.predict(self.X_test)
            y_true = self.y_test
        else:
            y_pred = self.model.predict(self.X_val)
            y_true = self.y_val

        # Calculate metrics
        metrics = {
            'accuracy': accuracy_score(self.y_test, y_pred),
            'precision': precision_score(self.y_test, y_pred, zero_division=0),
            'recall': recall_score(self.y_test, y_pred, zero_division=0),
            'f1': f1_score(self.y_test, y_pred, zero_division=0)
        } 
        
        # Print metrics in a formatted table
        print("\nModel Performance Metrics:")
        print("=" * 40)
        print(f"{'Metric':<15} {'Score':>10}")
        print("-" * 40)
        for metric, score in metrics.items():
            print(f"{metric.capitalize():<15} {score:>10.3f}")
        print("=" * 40)
        
        # Print confusion matrix
        cm = confusion_matrix(self.y_test, y_pred)
        print("\nConfusion Matrix:")
        print("-" * 40)
        print("               Predicted")
        print("              Neg    Pos")
        print(f"Actual Neg    {cm[0][0]:<6} {cm[0][1]:<6}")
        print(f"      Pos    {cm[1][0]:<6} {cm[1][1]:<6}")
        print("-" * 40)
        
        # Calculate and print class-wise metrics
        print("\nPer-Class Metrics:")
        print("-" * 40)
        print(f"{'Class':<8} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        print("-" * 40)
        
        # Get per-class metrics
        prec_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        rec_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        for i, (p, r, f) in enumerate(zip(prec_per_class, rec_per_class, f1_per_class)):
            print(f"{i:<8} {p:>10.3f} {r:>10.3f} {f:>10.3f}")
        print("-" * 40)

        return metrics
    
    def gridSearch(self):
        param_grid = {
            'n_estimators': [50, 100, 200, 300, 500],
            'max_depth': [3, 5, 10, 15, None],
            'max_features': ['sqrt', 'log2', None, 0.5, 3],
        }

        scoring_metrics = {
            'accuracy': make_scorer(accuracy_score),
            'precision': make_scorer(precision_score, zero_division=0),
            'recall': make_scorer(recall_score, zero_division=0),
            'f1': make_scorer(f1_score, zero_division=0)
        }

        grid_search = GridSearchCV(
            RandomForestClassifier(random_state=42),
            param_grid,
            scoring=scoring_metrics,
            refit='f1',  # Refit on the model with the best F1 score
            cv=5,
            verbose=2,
            n_jobs=-1
        )

        grid_search.fit(self.X_train, self.y_train)
        print("Best parameters (based on F1):", grid_search.best_params_)

        # Save the best model found
        self.model = grid_search.best_estimator_

        return grid_search

    #pretty much repeat of other evaluation but more customizable inputs
    def _evaluate_specific_model(self, model, X, y):
        y_pred = model.predict(X)
        metrics = {
            'accuracy': accuracy_score(y, y_pred),
            'precision': precision_score(y, y_pred, zero_division=0),
            'recall': recall_score(y, y_pred, zero_division=0),
            'f1': f1_score(y, y_pred, zero_division=0)
        }
        print("-" * 40)
        for metric, score in metrics.items():
            print(f"{metric.capitalize():<10}: {score:.3f}")
        print("-" * 40)
        
        return metrics 

    # def evaluate_top_5_configs(self):
    #     param_grid = {
    #         'n_estimators': [50, 100, 200, 300, 500],
    #         'max_depth': [3, 5, 10, 15, None],
    #         'max_features': ['sqrt', 'log2', None, 0.5, 3],
    #     }

    #     scoring_metrics = {
    #         'accuracy': make_scorer(accuracy_score),
    #         'precision': make_scorer(precision_score, zero_division=0),
    #         'recall': make_scorer(recall_score, zero_division=0),
    #         'f1': make_scorer(f1_score, zero_division=0)
    #     }

    #     grid_search = GridSearchCV(
    #         RandomForestClassifier(random_state=42),
    #         param_grid,
    #         scoring=scoring_metrics,
    #         refit='f1',
    #         cv=5,
    #         verbose=2,
    #         n_jobs=-1
    #     )

    #     grid_search.fit(self.X_train, self.y_train)
    #     results_df = pd.DataFrame(grid_search.cv_results_)

    #     # Select and clean up the results table
    #     results_table = results_df[
    #         ['param_n_estimators', 'param_max_depth', 'param_max_features',
    #         'mean_test_accuracy', 'mean_test_precision', 'mean_test_recall', 'mean_test_f1']
    #     ].rename(columns={
    #         'param_n_estimators': 'n_estimators',
    #         'param_max_depth': 'max_depth',
    #         'param_max_features': 'max_features',
    #         'mean_test_accuracy': 'mean_accuracy',
    #         'mean_test_precision': 'mean_precision',
    #         'mean_test_recall': 'mean_recall',
    #         'mean_test_f1': 'mean_f1'
    #     })

    #     results_table[['mean_accuracy', 'mean_precision', 'mean_recall', 'mean_f1']] = results_table[
    #         ['mean_accuracy', 'mean_precision', 'mean_recall', 'mean_f1']
    #     ].round(3)

    #     # Select top 5 unique F1 configurations
    #     top_5 = results_table.drop_duplicates(subset=['mean_f1']).sort_values(by='mean_f1', ascending=False).head(5)
        
    #     print("\nTop 5 Unique Configurations by F1:\n", top_5)

    #     # Evaluate each configuration on val and test sets
    #     for idx, row in top_5.iterrows():
    #         print("\nEvaluating Configuration:")
    #         print(f"n_estimators: {row['n_estimators']}, max_depth: {row['max_depth']}, max_features: {row['max_features']}")
    #         # Create and train model with this configuration
    #         model = RandomForestClassifier(
    #             n_estimators=row['n_estimators'],
    #             max_depth=row['max_depth'],
    #             max_features=row['max_features'],
    #             random_state=42
    #         )
    #         model.fit(self.X_train, self.y_train)

    #         # Evaluate on validation set
    #         print("\nValidation Set Metrics:")
    #         self._evaluate_specific_model(model, self.X_val, self.y_val)

    #         # Evaluate on test set
    #         print("\nTest Set Metrics:")
    #         self._evaluate_specific_model(model, self.X_test, self.y_test)
    
    # def evaluate_configurations(self, estimator_configs, train_loader = self.train_loader, 
    #                                     val_loader = self.val_loader, test_loader = self.test_loader):

    #     records = []
    #     dataset = AlzheimerDataset('data')  # Load once for CDR mapping

    #     for config in estimator_configs:
    #         n_estimators, max_depth, max_features = config
    #         print(f"\nEvaluating Config: est={n_estimators}, depth={max_depth}, feat={max_features}")

    #         # Initialize and train model
    #         model = TabularBaseline(
    #             n_estimators=n_estimators, 
    #             max_depth=max_depth, 
    #             max_features=max_features
    #         )
    #         model.X_train, model.y_train = train_loader
    #         model.X_val, model.y_val = val_loader
    #         model.X_test, model.y_test = test_loader
    #         model.train()

    #         # Evaluate on train, val, and test sets
    #         for dataset_name, (X, y) in [('Train', (model.X_train, model.y_train)),
    #                                     ('Validation', (model.X_val, model.y_val)),
    #                                     ('Test', (model.X_test, model.y_test))]:
    #             metrics = model._evaluate_specific_model(model.model, X, y)
    #             record = {
    #                 'Set': dataset_name,
    #                 'n_estimators': n_estimators,
    #                 'max_depth': max_depth,
    #                 'max_features': max_features,
    #                 **metrics
    #             }
    #             records.append(record)

    #     return pd.DataFrame(records)


class MRIBaseline:
    '''
    Fine-tuned VGG-16 CNN for handling MRI data.

    This class is pretty much a disaster or methods that more or less do the same thing but 
    I changed as development progressed to suit my needs (whether for plotting purposes or running things
    faster)

    Overview of the methods:
        load_data -- given data loaders, stores appropriate MRI and CDR/labels
                    into class attributes
        _init_model -- just initializes the model adapated for grayscale MRI input
                       and binary classification
        _calculate_class_weights -- using training set, calculates weights to handle 
                                    class imbalance during training
        _get_training_componments -- sets up loiss funciton, optimizer and learning rate scheduler
                                     given a model and tensor of class weights
        _train_epoch -- trains a specific model for one epoch on a specific slice. returns avg loss over epoch
        _train_model_with_early_stopping -- same as ^^ but with early stopping to avoid overfitting. calls ^^
        train_on_specific_slice -- trains and saves best model on specific slice of mri. just made this to save time
        evaluate -- given a dataset, evaluates model on it and returns dictionary of metrics
        _plot_metrics -=- pltos bar chart of performance metrics across MRI slices (this should be the 1x4 plot with blue bars in the paper)
        
    '''
    def __init__(self):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model = None  # Will store the best single-slice model
        self.multi_slice_model = None  # Will store the multi-slice model
        self.slice_models = []

        self.train_data = {}
        self.y_train = None
        self.val_data = {}
        self.y_val = None
        self.test_data = {}
        self.y_test = None

    def load_data(self, train_loader, val_loader, test_loader):
        # Load training data
        self.train_data = []
        for batch in tqdm(train_loader, desc="Loading training data"):
            self.train_data.append({
                'mri': batch['mri'].to(self.device),
                'label': batch['label'].to(self.device).squeeze(),
                'cdr': batch['multiclass_label'].squeeze().cpu().tolist()  # Store CDR values directly
            })

        # Validation data
        self.val_data = []
        for batch in tqdm(val_loader, desc="Loading validation data"):
            self.val_data.append({
                'mri': batch['mri'].to(self.device),
                'label': batch['label'].to(self.device).squeeze(),
                'cdr': batch['multiclass_label'].squeeze().cpu().tolist()
            })

        # Test data
        self.test_data = []
        for batch in tqdm(test_loader, desc="Loading test data"):
            self.test_data.append({
                'mri': batch['mri'].to(self.device),
                'label': batch['label'].to(self.device).squeeze(),
                'cdr': batch['multiclass_label'].squeeze().cpu().tolist()
            })

        # Store label and CDR arrays directly for plotting/metrics
        self.y_train = torch.cat([batch['label'] for batch in train_loader]).tolist()
        self.y_val = torch.cat([batch['label'] for batch in val_loader]).tolist()
        self.y_test = torch.cat([batch['label'] for batch in test_loader]).tolist()

        self.cdr_train = torch.cat([batch['multiclass_label'] for batch in train_loader]).tolist()
        self.cdr_val = torch.cat([batch['multiclass_label'] for batch in val_loader]).tolist()
        self.cdr_test = torch.cat([batch['multiclass_label'] for batch in test_loader]).tolist()
        
    def _init_model(self):
        """Initialize a VGG16 model."""
        model = vgg16(weights=VGG16_Weights.DEFAULT)
        model.features[0] = nn.Conv2d(1, 64, kernel_size=3, padding=1)
        model.classifier[-1] = nn.Linear(4096, 2)
        model.features = nn.Sequential(
            model.features[0],
            nn.BatchNorm2d(64),
            *list(model.features.children())[1:]
        )
        return model.to(self.device)

    def _calculate_class_weights(self, train_loader):
        '''
        Our data is imbalanced, so we need to calculate class weights to prevent the model from being biased towards the majority class.
        '''
        label_counts = torch.zeros(2)
        for batch in train_loader:
            labels = batch['label'].to(self.device).long()
            label_counts += torch.bincount(labels.view(-1), minlength=2)
        return (label_counts.sum() - label_counts) / label_counts

    def _get_training_components(self, model, class_weights):
        criterion = nn.CrossEntropyLoss(weight=class_weights)
        optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2)
        return criterion, optimizer, scheduler

    def _train_epoch(self, model, loader, criterion, optimizer, slice_idx):
        model.train()
        epoch_loss = 0

        for batch in loader:

            optimizer.zero_grad()
            slice_imgs = batch['mri'][:, :, slice_idx, :, :].to(self.device)
            labels = batch['label'].view(-1).to(self.device).long()
            outputs = model(slice_imgs)
            loss = criterion(outputs, labels)
            loss += 0.01 * sum(p.pow(2).sum() for p in model.parameters())  # L2 reg    
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            epoch_loss += loss.item()
        
        return epoch_loss / len(loader)
    
    def _train_model_with_early_stopping(self, model, train_loader, val_loader, criterion, optimizer, scheduler, slice_idx, max_epochs=20, patience=7):
        best_model, best_val_f1 = None, 0
        patience_counter = 0
        for epoch in tqdm(range(max_epochs), desc="Training with early stopping"):
            avg_loss = self._train_epoch(model, train_loader, criterion, optimizer, slice_idx)
            val_metrics = self.evaluate(model, val_loader, slice_idx)
            val_f1 = val_metrics['f1']
            print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, val_f1={val_f1:.3f}")
            scheduler.step(val_f1)
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model = copy.deepcopy(model)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= patience:
                    print("  Early stopping triggered")
                    break
        return best_model, best_val_f1

    def evaluate(self, model, data_loader, slice_idx):
        model.eval()
        all_preds, all_labels, all_probs = [], [], []
        
        with torch.no_grad():
            for batch in data_loader:
                if slice_idx is not None and isinstance(slice_idx, int):
                    # Single-slice model
                    slice_imgs = batch['mri'][:, :, slice_idx, :, :].to(self.device)
                else:
                    # Multi-slice model - use all slices
                    slice_imgs = batch['mri'].to(self.device)
                    slice_imgs = slice_imgs.squeeze(1)  # Remove the extra channel dimension
                
                labels = batch['label'].to(self.device).long()
                
                outputs = model(slice_imgs)
                probs = F.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                all_probs.extend(probs.cpu().tolist())
                all_preds.extend(predicted.cpu().tolist())
                all_labels.extend(labels.cpu().tolist())
        
        # Calculate all metrics
        metrics = {
            'accuracy': accuracy_score(all_labels, all_preds),
            'precision': precision_score(all_labels, all_preds, zero_division=0),
            'recall': recall_score(all_labels, all_preds, zero_division=0),
            'f1': f1_score(all_labels, all_preds, zero_division=0),
            'confusion_matrix': confusion_matrix(all_labels, all_preds),
            'probabilities': np.array(all_probs)
        }
        
        # Add per-class metrics
        metrics.update({
            'precision_per_class': precision_score(all_labels, all_preds, average=None, zero_division=0),
            'recall_per_class': recall_score(all_labels, all_preds, average=None, zero_division=0),
            'f1_per_class': f1_score(all_labels, all_preds, average=None, zero_division=0)
        })
        
        # # Printing Stuff
        # # Print detailed metrics
        # print("\nDetailed Evaluation Metrics:")
        # print("=" * 50)
        
        # # Overall metrics
        # print("Overall Metrics:")
        # print("-" * 50)
        # print(f"{'Metric':<15} {'Score':>10}")
        # print("-" * 50)
        # for metric in ['accuracy', 'precision', 'recall', 'f1']:
        #     print(f"{metric.capitalize():<15} {metrics[metric]:>10.3f}")
        # print("-" * 50)
        
        # # Confusion matrix
        # cm = metrics['confusion_matrix']
        # print("\nConfusion Matrix:")
        # print("-" * 50)
        # print("               Predicted")
        # print("              Neg    Pos")
        # print(f"Actual Neg    {cm[0][0]:<6} {cm[0][1]:<6}")
        # print(f"      Pos    {cm[1][0]:<6} {cm[1][1]:<6}")
        # print("-" * 50)
        
        # # Per-class metrics
        # print("\nPer-Class Metrics:")
        # print("-" * 50)
        # print(f"{'Class':<8} {'Precision':>10} {'Recall':>10} {'F1':>10}")
        # print("-" * 50)
        # for i in range(2):
        #     print(f"{i:<8} {metrics['precision_per_class'][i]:>10.3f} "
        #           f"{metrics['recall_per_class'][i]:>10.3f} "
        #           f"{metrics['f1_per_class'][i]:>10.3f}")
        # print("-" * 50)
        
        # # Prediction probabilities
        # print("\nAverage Prediction Probabilities:")
        # print("-" * 50)
        # print(f"{'Class':<8} {'Probability':>12}")
        # print("-" * 50)
        # avg_probs = np.mean(metrics['probabilities'], axis=0)
        # for i, prob in enumerate(avg_probs):
        #     print(f"{i:<8} {prob:>12.3f}")
        # print("-" * 50)
        
        return metrics

    def _plot_metrics(self, per_slice_metrics, best_model_idx):
        """Plot performance metrics across slices."""
        plt.figure(figsize=(15, 5))
        for i, metric in enumerate(['accuracy', 'precision', 'recall', 'f1']):
            plt.subplot(1, 4, i+1)
            values = [m[metric] for m in per_slice_metrics]
            plt.bar(range(len(values)), values)
            plt.axhline(y=np.mean(values), color='r', linestyle='--', label='Mean')
            if metric == 'f1':
                plt.axvline(x=best_model_idx, color='g', linestyle='--', label='Best')
            plt.xlabel('Slice')
            plt.ylabel(metric.capitalize())
            plt.title(f'{metric.capitalize()} per Slice')
            plt.legend()
        plt.tight_layout()
        plt.savefig('mri_per_slice_model_performance.png')
        plt.close()

    def train_on_specific_slice(self, train_loader, val_loader, test_loader, slice_idx):
        print(f"Training model for slice {slice_idx}...")
        class_weights = self._calculate_class_weights(train_loader).to(self.device)
        model = self._init_model()
        criterion, optimizer, scheduler = self._get_training_components(model, class_weights)

        best_model, best_val_f1 = None, 0
        patience_counter = 0
        training_history = []

        for epoch in tqdm(range(20), desc="Training with early stopping"):
            avg_loss = self._train_epoch(model, train_loader, criterion, optimizer, slice_idx)
            val_metrics = self.evaluate(model, val_loader, slice_idx)
            val_f1 = val_metrics['f1']
            print(f"  Epoch {epoch+1}: loss={avg_loss:.4f}, val_f1={val_f1:.3f}")
            scheduler.step(val_f1)

            # Store metrics for plotting
            training_history.append({
                'epoch': epoch + 1,
                'loss': avg_loss,
                'accuracy': val_metrics['accuracy'],
                'f1': val_f1
            })

            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                best_model = copy.deepcopy(model)
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 7:
                    print("  Early stopping triggered")
                    break

        # Save model and return training history
        self.model = copy.deepcopy(best_model)
        return best_model, training_history


    def _plot_training_history(self, history):
        """Plot training history metrics."""
        plt.figure(figsize=(15, 5))
        plt.subplot(1, 3, 1)
        plt.plot([h['loss'] for h in history], label='Loss')
        plt.xlabel('Epoch')
        plt.ylabel('Loss')
        plt.title('Training Loss')
        plt.legend()
        
        # Plot accuracy
        plt.subplot(1, 3, 2)
        plt.plot([h['accuracy'] for h in history], label='Accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.title('Validation Accuracy')
        plt.legend()
        
        # Plot F1 score
        plt.subplot(1, 3, 3)
        plt.plot([h['f1'] for h in history], label='F1')
        plt.xlabel('Epoch')
        plt.ylabel('F1 Score')
        plt.title('Validation F1 Score')
        plt.legend()
        
        plt.tight_layout()
        plt.close()
    

    # 9 input channel model stuff -- not using obvs
    # def _init_multi_slice_model(self):
    #     """Initialize a VGG16 model modified to accept 9 input channels (all slices)."""
    #     model = vgg16(weights=VGG16_Weights.DEFAULT)
    #     # Modify first layer to accept 9 channels instead of 3
    #     model.features[0] = nn.Conv2d(9, 64, kernel_size=3, padding=1)
    #     model.classifier[-1] = nn.Linear(4096, 2)
        
    #     # Add batch normalization
    #     model.features = nn.Sequential(
    #         model.features[0],
    #         nn.BatchNorm2d(64),
    #         *list(model.features.children())[1:]
    #     )
    #     return model.to(self.device)

    # def train_multi_slice_model(self, train_loader, test_loader, num_epochs=10):
    #     """Train a model using all 9 slices together."""
    #     print("\nTraining model with all slices...")
        
    #     # Initialize model and training components
    #     model = self._init_multi_slice_model()
        
    #     # Calculate class weights
    #     class_weights = self._calculate_class_weights(train_loader)
    #     print(f"Class weights: {class_weights}")
        
    #     criterion = nn.CrossEntropyLoss(weight=class_weights.to(self.device))
    #     optimizer = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
    #     scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.1, patience=2)
        
    #     best_val_f1 = 0
    #     patience_counter = 0
    #     training_history = []
    #     best_model = None
        
    #     for epoch in range(num_epochs):
    #         # Training phase
    #         model.train()
    #         epoch_loss = 0
            
    #         for batch in tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}"):
    #             optimizer.zero_grad()
                
    #             # Reshape MRI data from [batch, 1, 9, H, W] to [batch, 9, H, W]
    #             mri = batch['mri'].to(self.device)
    #             mri = mri.squeeze(1)  # Remove the extra channel dimension
    #             labels = batch['label'].view(-1).to(self.device).long()
                
    #             outputs = model(mri)
    #             loss = criterion(outputs, labels)
                
    #             # Add L2 regularization
    #             loss += 0.01 * sum(p.pow(2).sum() for p in model.parameters())
                
    #             loss.backward()
    #             torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    #             optimizer.step()
                
    #             epoch_loss += loss.item()
            
    #         avg_loss = epoch_loss / len(train_loader)
            
    #         # Validation phase
    #         model.eval()
    #         val_preds, val_labels = [], []
            
    #         with torch.no_grad():
    #             for batch in test_loader:
    #                 mri = batch['mri'].to(self.device)
    #                 mri = mri.squeeze(1)  # Remove the extra channel dimension
    #                 labels = batch['label'].to(self.device).long()
                    
    #                 outputs = model(mri)
    #                 _, predicted = torch.max(outputs, 1)
    #                 val_preds.extend(predicted.cpu().tolist())
    #                 val_labels.extend(labels.cpu().tolist())
            
    #         # Calculate validation metrics
    #         val_metrics = {
    #             'accuracy': accuracy_score(val_labels, val_preds),
    #             'precision': precision_score(val_labels, val_preds, zero_division=0),
    #             'recall': recall_score(val_labels, val_preds, zero_division=0),
    #             'f1': f1_score(val_labels, val_preds, zero_division=0)
    #         }
            
    #         print(f"\nEpoch {epoch+1} metrics:")
    #         print(f"Loss: {avg_loss:.4f}")
    #         print(f"Val Accuracy: {val_metrics['accuracy']:.3f}")
    #         print(f"Val F1: {val_metrics['f1']:.3f}")
            
    #         # Learning rate scheduling
    #         scheduler.step(val_metrics['f1'])
            
    #         # Early stopping check
    #         if val_metrics['f1'] > best_val_f1:
    #             best_val_f1 = val_metrics['f1']
    #             patience_counter = 0
    #             best_model = copy.deepcopy(model)
    #         else:
    #             patience_counter += 1
    #             if patience_counter >= 5:
    #                 print("Early stopping triggered")
    #                 break
            
    #         training_history.append({
    #             'epoch': epoch + 1,
    #             'loss': avg_loss,
    #             **val_metrics
    #         })
        
    #     # Store the best multi-slice model
    #     self.multi_slice_model = best_model
    #     print("Stored multi-slice model")
        
    #     # Final evaluation
    #     print("\nFinal Evaluation:")
    #     final_metrics = self.evaluate(best_model, test_loader, slice_idx=None)
        
    #     # Plot training history
    #     self._plot_training_history(training_history)
        
    #     return {
    #         'training_history': training_history,
    #         'final_metrics': final_metrics
    #     }

    # automatically do all 9 slice evaluations
    # def train_and_evaluate_per_slice_models(self, train_loader, test_loader, num_slices):
    #     print("Calculating class weights...")
    #     class_weights = self._calculate_class_weights(train_loader)
    #     print(f"Class weights: {class_weights}")

    #     best_model_idx, best_f1 = 0, 0
    #     per_slice_metrics, slice_metrics = [], []

    #     for slice_idx in tqdm(range(num_slices), desc="Training slice models"):
    #         model = self._init_model()
    #         criterion, optimizer, scheduler = self._get_training_components(model, class_weights)


    #         model, val_f1 = self._train_model_with_early_stopping(
    #             model, train_loader, test_loader, criterion, optimizer, scheduler, slice_idx
    #         )

    #         metrics = self.evaluate(model, test_loader, slice_idx)
    #         per_slice_metrics.append(metrics)
    #         self.slice_models.append(model)
            
    #         print(f"Slice {slice_idx} metrics: "
    #               f"Acc={metrics['accuracy']:.3f} "
    #               f"F1={metrics['f1']:.3f}")
            
    #         if metrics['f1'] > best_f1:
    #             best_f1 = metrics['f1']
    #             best_model_idx = slice_idx
    #             self.model = copy.deepcopy(model)  # Store best model

    #     print(f"\nSelected model trained on slice {best_model_idx} with F1={best_f1:.3f}")
    #     self._plot_metrics(per_slice_metrics, best_model_idx)
    #     return per_slice_metrics
        
    # def apply_gradcam_to_mri(self, device, test_loader,num_patients=5, save_prefix='gradcam_patient'):

    #     model = self.model

    #     model.eval()
    #     # Pick 5 random indices from the test set
    #     all_indices = list(range(len(test_loader.dataset)))
    #     selected_indices = random.sample(all_indices, min(num_patients, len(all_indices)))
        
    #     # Get the last convolutional layer name for VGG16
    #     target_layer = model.features[-1]
        
    #     for i, idx in enumerate(selected_indices):
    #         # Get the sample
    #         sample = test_loader.dataset[idx]
    #         mri = sample['mri'][:, 1:2, :, :].to(device)  # Use only slice 1, shape: [1, 1, 224, 224] 
    #         label = sample['label'].item()
            
    #         # Forward pass
    #         mri = mri.unsqueeze(0) if mri.dim() == 3 else mri  # Ensure batch dimension
    #         mri.requires_grad = True
            
    #         # Register hook to get gradients and activations
    #         activations = []
    #         gradients = []
    #         def forward_hook(module, input, output):
    #             activations.append(output)
    #         def backward_hook(module, grad_in, grad_out):
    #             gradients.append(grad_out[0])
    #         handle_fwd = target_layer.register_forward_hook(forward_hook)
    #         handle_bwd = target_layer.register_backward_hook(backward_hook)
            
    #         # Forward and backward
    #         output = model(mri)
    #         pred_class = output.argmax(dim=1).item()
    #         score = output[0, pred_class]
    #         model.zero_grad()
    #         score.backward()
            
    #         # Get hooked data
    #         acts = activations[0].detach().cpu().numpy()[0]
    #         grads = gradients[0].detach().cpu().numpy()[0]
    #         weights = grads.mean(axis=(1, 2))  # Global average pooling
            
    #         # Compute Grad-CAM
    #         cam = np.zeros(acts.shape[1:], dtype=np.float32)
    #         for k, w in enumerate(weights):
    #             cam += w * acts[k]
    #         cam = np.maximum(cam, 0)
    #         cam = cam / (cam.max() + 1e-8)
    #         cam = np.uint8(255 * cam)
    #         cam = np.resize(cam, (224, 224))
            
    #         # Overlay on MRI slice
    #         mri_img = mri[0, 0].detach().cpu().numpy()
    #         plt.figure(figsize=(6, 6))
    #         plt.imshow(mri_img, cmap='gray')
    #         plt.imshow(cam, cmap='jet', alpha=0.5)
    #         plt.title(f'Patient {idx} | True label: {label} | Pred: {pred_class}')
    #         plt.axis('off')
    #         plt.tight_layout()
    #         plt.savefig(f'{save_prefix}_{i+1}.png')
    #         plt.close()
            
    #         # Remove hooks
    #         handle_fwd.remove()
    #         handle_bwd.remove()
            
    #     print(f"Saved Grad-CAM visualizations for {num_patients} patients.")

    # def compare_models(self, single_slice_metrics, multi_slice_metrics):
    #     """Compare performance between single-slice and multi-slice models."""
    #     print("\nModel Comparison:")
    #     print("=" * 60)
        
    #     # Get best single-slice performance
    #     best_single_slice = max(single_slice_metrics, key=lambda x: x['f1'])
    #     best_slice_idx = single_slice_metrics.index(best_single_slice)
        
    #     # Prepare comparison table
    #     print(f"{'Metric':<15} {'Best Single-Slice':>20} {'Multi-Slice':>20}")
    #     print("-" * 60)
        
    #     metrics_to_compare = ['accuracy', 'precision', 'recall', 'f1']
    #     for metric in metrics_to_compare:
    #         single_value = best_single_slice[metric]
    #         multi_value = multi_slice_metrics['final_metrics'][metric]
    #         diff = multi_value - single_value
    #         diff_str = f"({'+' if diff >= 0 else ''}{diff:.3f})"
    #         print(f"{metric.capitalize():<15} {single_value:>20.3f} {multi_value:>16.3f} {diff_str:>8}")
        
    #     print("-" * 60)
    #     print(f"Best single-slice model used slice {best_slice_idx}")
        
    #     # Plot comparison
    #     plt.figure(figsize=(10, 6))
    #     x = np.arange(len(metrics_to_compare))
    #     width = 0.35
        
    #     single_scores = [best_single_slice[m] for m in metrics_to_compare]
    #     multi_scores = [multi_slice_metrics['final_metrics'][m] for m in metrics_to_compare]
        
    #     plt.bar(x - width/2, single_scores, width, label='Best Single-Slice')
    #     plt.bar(x + width/2, multi_scores, width, label='Multi-Slice')
        
    #     plt.xlabel('Metrics')
    #     plt.ylabel('Score')
    #     plt.title('Single-Slice vs Multi-Slice Model Performance')
    #     plt.xticks(x, [m.capitalize() for m in metrics_to_compare])
    #     plt.legend()
        
    #     plt.tight_layout()
    #     plt.savefig('model_comparison.png')
    #     plt.close()
    
# def debug_slices(self, loader):
#     for batch in loader:
#         mri = batch['mri']
#         print(f"MRI batch shape: {mri.shape}")
#         num_slices = mri.shape[2]
#         for slice_idx in range(num_slices):
#             if mri.dim() == 5:
#                 slice_img = mri[0, :, slice_idx, :, :].cpu().numpy()
#             else:
#                 slice_img = mri[0, :, slice_idx, :].cpu().numpy()
#             # Print a hash of the slice to check uniqueness
#             print(f"Slice {slice_idx} hash: {hash(slice_img.tobytes())}")
#         break  # Only check the first batch
