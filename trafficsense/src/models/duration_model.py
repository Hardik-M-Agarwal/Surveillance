import pandas as pd
import numpy as np
import json
import joblib
import os
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error, mean_squared_error
from lightgbm import LGBMRegressor

class DurationRegressor:
    def __init__(self):
        self.model = LGBMRegressor(
            n_estimators=500,
            max_depth=8,
            learning_rate=0.03,
            num_leaves=63,
            subsample=0.8,
            colsample_bytree=0.8,
            reg_alpha=0.1,
            reg_lambda=0.1,
            random_state=42
        )
        
    def prepare_data(self, df):
        print("Preparing data for duration model...")
        # Filter valid durations and non-infrastructure
        df = df[df['duration_minutes'].notna()]
        df = df[df['duration_minutes'] > 0]
        if 'is_infrastructure' in df.columns:
            df = df[df['is_infrastructure'] == False]
            
        with open('models/feature_columns.json', 'r') as f:
            feature_columns = json.load(f)
            
        X = df[feature_columns].fillna(0)
        y = np.log1p(df['duration_minutes'])
        
        # Stratified split on event_cause to preserve distribution
        # Need to handle rare classes that might have only 1 instance
        counts = df['event_cause'].value_counts()
        valid_causes = counts[counts > 1].index
        
        stratify_col = df['event_cause'].copy()
        stratify_col[~stratify_col.isin(valid_causes)] = 'others'
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, stratify=stratify_col, random_state=42
        )
        
        os.makedirs('data/processed/train_test_splits', exist_ok=True)
        X_train.to_csv('data/processed/train_test_splits/X_train_duration.csv', index=False)
        X_test.to_csv('data/processed/train_test_splits/X_test_duration.csv', index=False)
        pd.Series(y_train).to_csv('data/processed/train_test_splits/y_train_duration.csv', index=False)
        pd.Series(y_test).to_csv('data/processed/train_test_splits/y_test_duration.csv', index=False)
        
        return X_train, X_test, y_train, y_test, df.loc[X_test.index, 'event_cause']

    def train(self, X_train, y_train):
        print("Training duration model...")
        X_t, X_v, y_t, y_v = train_test_split(
            X_train, y_train, test_size=0.1, random_state=42
        )
        # Note: lightgbm early_stopping_rounds parameter is passed to fit via callbacks in newer versions
        # or directly if supported. We'll use the scikit-learn API approach
        from lightgbm.callback import early_stopping
        self.model.fit(
            X_t, y_t,
            eval_set=[(X_v, y_v)],
            callbacks=[early_stopping(stopping_rounds=50)]
        )

    def evaluate(self, X_test, y_test, test_causes):
        print("Evaluating duration model...")
        y_pred_log = self.model.predict(X_test)
        
        y_pred = np.expm1(y_pred_log)
        y_actual = np.expm1(y_test)
        
        mae = mean_absolute_error(y_actual, y_pred)
        rmse = np.sqrt(mean_squared_error(y_actual, y_pred))
        
        print(f"Overall MAE: {mae:.2f} minutes")
        print(f"Overall RMSE: {rmse:.2f} minutes")
        
        # Plot
        plt.figure(figsize=(8,6))
        plt.scatter(y_actual, y_pred, alpha=0.5)
        plt.plot([min(y_actual), max(y_actual)], [min(y_actual), max(y_actual)], 'r--')
        plt.xlabel('Actual Duration (min)')
        plt.ylabel('Predicted Duration (min)')
        plt.title('Actual vs Predicted Duration (Original Scale)')
        
        os.makedirs('models', exist_ok=True)
        plt.savefig('models/duration_scatter.png')
        plt.close()
        
        # MAE per event cause
        results = pd.DataFrame({'Actual': y_actual, 'Predicted': y_pred, 'Cause': test_causes.values})
        results['Error'] = np.abs(results['Actual'] - results['Predicted'])
        print("\nMAE per event_cause group:")
        print(results.groupby('Cause')['Error'].mean().sort_values())
        
        return mae, rmse

    def predict(self, X):
        return np.expm1(self.model.predict(X))

    def save(self, path='models/duration_model.pkl'):
        print(f"Saving duration model to {path}...")
        joblib.dump(self.model, path)

    def load(self, path='models/duration_model.pkl'):
        print(f"Loading duration model from {path}...")
        self.model = joblib.load(path)
