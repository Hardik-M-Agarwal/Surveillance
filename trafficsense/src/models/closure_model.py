import pandas as pd
import numpy as np
import json
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, precision_recall_curve, roc_auc_score
from imblearn.over_sampling import SMOTE
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier

class ClosureClassifier:
    def __init__(self):
        self.baseline_model = LogisticRegression(max_iter=1000, random_state=42)
        self.xgb_model = XGBClassifier(random_state=42)
        # We will set scale_pos_weight in train()
        
    def prepare_data(self, df):
        print("Preparing data for closure model...")
        with open('models/feature_columns.json', 'r') as f:
            feature_columns = json.load(f)
            
        features = [col for col in feature_columns if col != 'requires_road_closure']
        
        X = df[features]
        y = df['requires_road_closure'].astype(int)
        
        # Fill any remaining NaNs with 0 to prevent SMOTE and LogisticRegression from crashing
        X = X.fillna(0)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=y, random_state=42
        )
        
        smote = SMOTE(random_state=42, sampling_strategy=0.3)
        X_train_sm, y_train_sm = smote.fit_resample(X_train, y_train)
        
        os.makedirs('data/processed/train_test_splits', exist_ok=True)
        X_train.to_csv('data/processed/train_test_splits/X_train_closure.csv', index=False)
        X_test.to_csv('data/processed/train_test_splits/X_test_closure.csv', index=False)
        y_train.to_csv('data/processed/train_test_splits/y_train_closure.csv', index=False)
        y_test.to_csv('data/processed/train_test_splits/y_test_closure.csv', index=False)
        
        return X_train_sm, X_test, y_train_sm, y_test

    def train(self, X_train, y_train):
        print("Training closure model...")
        self.baseline_model.fit(X_train, y_train)
        
        scale_pos_weight = (len(y_train) - sum(y_train)) / sum(y_train)
        self.xgb_model.set_params(scale_pos_weight=scale_pos_weight)
        self.xgb_model.fit(X_train, y_train)

    def evaluate(self, X_test, y_test, threshold=0.35):
        print(f"Evaluating closure model with threshold {threshold}...")
        y_prob = self.xgb_model.predict_proba(X_test)[:, 1]
        y_pred = (y_prob >= threshold).astype(int)
        
        print("Classification Report (Thresholded for Recall):")
        print(classification_report(y_test, y_pred))
        
        roc_auc = roc_auc_score(y_test, y_prob)
        print(f"ROC-AUC: {roc_auc:.4f}")
        
        # PR Curve
        precision, recall, thresholds = precision_recall_curve(y_test, y_prob)
        plt.figure(figsize=(8,6))
        plt.plot(recall, precision, marker='.')
        plt.xlabel('Recall')
        plt.ylabel('Precision')
        plt.title('Precision-Recall Curve (Closure Model)')
        
        os.makedirs('models', exist_ok=True)
        plt.savefig('models/closure_pr_curve.png')
        plt.close()
        
        return roc_auc

    def predict_proba(self, X):
        return self.xgb_model.predict_proba(X)[:, 1]

    def save(self, path='models/closure_model.pkl'):
        print(f"Saving closure models to {path}...")
        joblib.dump({
            'baseline': self.baseline_model,
            'xgb': self.xgb_model
        }, path)

    def load(self, path='models/closure_model.pkl'):
        print(f"Loading closure models from {path}...")
        data = joblib.load(path)
        self.baseline_model = data['baseline']
        self.xgb_model = data['xgb']
