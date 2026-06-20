import shap
import pandas as pd
import numpy as np

class SHAPExplainer:
    def __init__(self):
        pass

    def explain_severity(self, model, X_instance, feature_names):
        # model is expected to be XGBClassifier
        explainer = shap.TreeExplainer(model)
        # TreeExplainer on multiclass XGBoost returns a list of arrays or an array of shape (n_samples, n_features, n_classes)
        shap_values = explainer.shap_values(X_instance)
        
        # Determine the predicted class for this instance
        pred_class = model.predict(X_instance)[0]
        
        if isinstance(shap_values, list):
            class_shap = shap_values[pred_class][0]
        else:
            if len(shap_values.shape) == 3:
                class_shap = shap_values[0, :, pred_class]
            else:
                class_shap = shap_values[0]
                
        # Get top 5 features
        top_indices = np.argsort(np.abs(class_shap))[-5:][::-1]
        
        explanation = []
        for idx in top_indices:
            val = class_shap[idx]
            explanation.append({
                'feature': feature_names[idx],
                'impact': float(val),
                'direction': 'increases severity' if val > 0 else 'decreases severity'
            })
            
        return explanation

    def explain_closure(self, model, X_instance, feature_names):
        # model is expected to be XGBClassifier
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X_instance)
        
        if isinstance(shap_values, list):
            val_array = shap_values[1][0] if len(shap_values) > 1 else shap_values[0][0]
        else:
            if len(shap_values.shape) == 3:
                val_array = shap_values[0, :, 1]
            elif len(shap_values.shape) == 2:
                val_array = shap_values[0]
            else:
                val_array = shap_values
                
        top_indices = np.argsort(np.abs(val_array))[-5:][::-1]
        
        explanation = []
        for idx in top_indices:
            val = val_array[idx]
            explanation.append({
                'feature': feature_names[idx],
                'impact': float(val),
                'direction': 'increases risk' if val > 0 else 'decreases risk'
            })
            
        return explanation

    def generate_explanation_text(self, shap_values, prediction_label):
        # shap_values is the list of dicts from above
        top_features = []
        for sv in shap_values[:3]:
            # Simple formatting
            feat_name = sv['feature'].replace('_encoded', '').replace('_', ' ')
            sign = "+" if sv['impact'] > 0 else ""
            top_features.append(f"{feat_name} ({sign}{sv['impact']:.2f})")
            
        reasons = ", ".join(top_features[:-1]) + f", and {top_features[-1]}" if len(top_features) > 1 else top_features[0]
        
        return f"This event is predicted **{prediction_label}** primarily because of these factors: {reasons}."
