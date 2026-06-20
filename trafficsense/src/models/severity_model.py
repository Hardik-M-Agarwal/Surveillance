import pandas as pd
import numpy as np
import json
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, f1_score, confusion_matrix
from sklearn.utils.class_weight import compute_sample_weight
from xgboost import XGBClassifier
from sklearn.preprocessing import LabelEncoder

class SeverityClassifier:
    def __init__(self):
        self.model = XGBClassifier(
            n_estimators=300,
            max_depth=6,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            scale_pos_weight=None,
            use_label_encoder=False,
            eval_metric='mlogloss',
            random_state=42
        )
        self.label_encoder = LabelEncoder()
        
    def prepare_data(self, df):
        print("Preparing data for severity model...")
        with open('models/feature_columns.json', 'r') as f:
            feature_columns = json.load(f)
            
        features = [col for col in feature_columns if col != 'requires_road_closure']
        
        # Ensure only known severity classes are kept, although dataset should have only 4
        # Drop any nan severity
        df = df.dropna(subset=['severity']).copy()
        
        X = df[features].fillna(0)
        y = df['severity']
        
        # Encode severity labels to integers for XGBoost
        y_encoded = self.label_encoder.fit_transform(y)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=0.2, stratify=y_encoded, random_state=42
        )
        
        os.makedirs('data/processed/train_test_splits', exist_ok=True)
        X_train.to_csv('data/processed/train_test_splits/X_train_severity.csv', index=False)
        X_test.to_csv('data/processed/train_test_splits/X_test_severity.csv', index=False)
        pd.Series(y_train).to_csv('data/processed/train_test_splits/y_train_severity.csv', index=False)
        pd.Series(y_test).to_csv('data/processed/train_test_splits/y_test_severity.csv', index=False)
        
        return X_train, X_test, y_train, y_test

    def train(self, X_train, y_train):
        print("Training severity model...")
        sample_weights = compute_sample_weight('balanced', y_train)
        self.model.fit(X_train, y_train, sample_weight=sample_weights)

    def evaluate(self, X_test, y_test):
        print("Evaluating severity model...")
        y_pred = self.model.predict(X_test)
        
        target_names = self.label_encoder.inverse_transform(np.unique(y_test))
        report = classification_report(y_test, y_pred, target_names=target_names)
        print("Classification Report:\n", report)
        
        macro_f1 = f1_score(y_test, y_pred, average='macro')
        print(f"Macro F1 Score: {macro_f1:.4f}")
        
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(8,6))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
        plt.ylabel('Actual')
        plt.xlabel('Predicted')
        plt.title('Severity Confusion Matrix')
        
        os.makedirs('models', exist_ok=True)
        plt.savefig('models/severity_confusion_matrix.png')
        plt.close()
        
        return macro_f1, report

    def save(self, path='models/severity_model.pkl'):
        print(f"Saving severity model to {path}...")
        joblib.dump({
            'model': self.model,
            'label_encoder': self.label_encoder
        }, path)

    def load(self, path='models/severity_model.pkl'):
        print(f"Loading severity model from {path}...")
        data = joblib.load(path)
        self.model = data['model']
        self.label_encoder = data['label_encoder']

    def predict(self, X):
        y_pred_encoded = self.model.predict(X)
        return self.label_encoder.inverse_transform(y_pred_encoded)
