# -*- coding: UTF-8 -*-
import numpy as np


def evaluate(acc_scores, f1_scores, rocauc_scores, gmean_scores, mcc_scores):
    return np.mean(acc_scores), np.std(acc_scores), np.mean(f1_scores), np.std(f1_scores), np.mean(
        rocauc_scores), np.std(rocauc_scores), np.mean(gmean_scores), np.std(gmean_scores), np.mean(mcc_scores), np.std(
        mcc_scores)


def evaluate_metric(y_true_train, y_pred_train):
    from sklearn.metrics import accuracy_score, f1_score, roc_auc_score, matthews_corrcoef, confusion_matrix

    accuracy_train = accuracy_score(y_true_train, y_pred_train)
    f1_train = f1_score(y_true_train, y_pred_train)
    auc_roc_train = roc_auc_score(y_true_train, y_pred_train)
    mcc_train = matthews_corrcoef(y_true_train, y_pred_train)
    tn, fp, fn, tp = confusion_matrix(y_true_train, y_pred_train).ravel()
    gmean_train = np.sqrt((tp / (tp + fn)) * (tn / (tn + fp)))

    return accuracy_train, f1_train, auc_roc_train, gmean_train, mcc_train
