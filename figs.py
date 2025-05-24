from dataloader import *
from unimodal_baseline import *
#from experiment import *

import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm
import random
import copy
import inspect
# import shap

import torch 
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Subset
from torchvision import models
from torchvision.models import vgg16, VGG16_Weights

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, precision_recall_curve, average_precision_score


###########GENERAL PLOTTING FUNCTIONS#############

train_loader, val_loader, test_loader = get_data_loaders('data')
dataset = AlzheimerDataset('data')

def plotByCDR(model, filename = None):
    y_pred = model.predict(model.X_test)
    y_true = model.y_test
    test_indices = test_loader.dataset.indices
    test_cdrs = np.array([dataset.multiclass_labels[i] for i in test_indices])

    # Create masks
    correct = (y_pred == y_true)
    incorrect = ~correct

    # Get CDR values for correct and incorrect predictions
    correct_cdrs = test_cdrs[correct]
    incorrect_cdrs = test_cdrs[incorrect]

    # Get all unique CDR values across both sets
    all_cdrs = np.unique(test_cdrs)

    # Count correct and incorrect per CDR
    correct_counts = np.array([np.sum(correct_cdrs == cdr) for cdr in all_cdrs])
    incorrect_counts = np.array([np.sum(incorrect_cdrs == cdr) for cdr in all_cdrs])

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    bar1 = ax.bar(all_cdrs, correct_counts, width=0.1, label='Correct', color='green')
    bar2 = ax.bar(all_cdrs, incorrect_counts, width=0.1, bottom=correct_counts, label='Incorrect', color='red')

    ax.set_xlabel('CDR Value')
    ax.set_ylabel('Number of Predictions')
    ax.set_title('Prediction Outcomes by CDR')
    ax.legend()
    plt.show()

import numpy as np
import matplotlib.pyplot as plt
import torch

def plotByCDR_torch(model, test_loader, dataset, slice_idx=6, filename=None):

    model.eval()
    y_pred = []
    y_true = []
    # Get the indices of the test set (for CDR lookup)
    if hasattr(test_loader.dataset, 'indices'):
        test_indices = test_loader.dataset.indices
    else:
        test_indices = np.arange(len(test_loader.dataset))
    test_cdrs = np.array([dataset.multiclass_labels[i] for i in test_indices])

    with torch.no_grad():
        for batch in test_loader:
            img = batch['mri'][:, :, slice_idx, :, :]  # [batch, 1, H, W]
            img = img.to(next(model.parameters()).device)
            outputs = model(img)
            if isinstance(outputs, dict):
                outputs = outputs['logits']
            predictions = torch.argmax(outputs, dim=1)
            y_pred.extend(predictions.cpu().numpy())
            y_true.extend(batch['label'].cpu().numpy())

    y_pred = np.array(y_pred).flatten()
    y_true = np.array(y_true).flatten()
    correct = (y_pred == y_true)
    incorrect = ~correct
    correct_cdrs = test_cdrs[correct]
    incorrect_cdrs = test_cdrs[incorrect]

    # Get all unique CDR values across both sets
    all_cdrs = np.unique(test_cdrs)

    # Count correct and incorrect per CDR
    correct_counts = np.array([np.sum(correct_cdrs == cdr) for cdr in all_cdrs])
    incorrect_counts = np.array([np.sum(incorrect_cdrs == cdr) for cdr in all_cdrs])
    total_counts = correct_counts + incorrect_counts

    # Plot
    fig, ax = plt.subplots(figsize=(7, 5))
    bar1 = ax.bar(all_cdrs, correct_counts, width=0.1, label='Correct', color='green')
    bar2 = ax.bar(all_cdrs, incorrect_counts, width=0.1, bottom=correct_counts, label='Incorrect', color='red')

    # Add percent correct above bars
    for i, (x, c, t) in enumerate(zip(all_cdrs, correct_counts, total_counts)):
        if t > 0:
            percent = 100 * c / t
            ax.text(x, c + incorrect_counts[i] + 0.5, f"{percent:.1f}%", ha='center', va='bottom', fontsize=10, fontweight='bold')

    ax.set_xlabel('CDR Value')
    ax.set_ylabel('Number of Predictions')
    ax.set_title('Prediction Outcomes by CDR')
    ax.legend()
    
    if filename:
        plt.savefig(filename)
    plt.show()