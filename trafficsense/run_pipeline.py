import argparse
import pandas as pd
from tqdm import tqdm
import time
from src.preprocessing import DataCleaner
from src.feature_engineering import FeatureEngineer
from src.models.severity_model import SeverityClassifier
from src.models.duration_model import DurationRegressor
from src.models.closure_model import ClosureClassifier
from src.recommender import ResourceRecommender

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--skip-train', action='store_true', help='Skip training and just load models')
    args = parser.parse_args()
    
    print("Starting TrafficSense Pipeline...")
    
    # 1. Load and clean data
    print("\n--- 1. Data Cleaning ---")
    cleaner = DataCleaner()
    df = cleaner.run_pipeline('data/raw/ps2.csv', 'data/processed/cleaned.csv')
    
    # 2. Feature engineering
    print("\n--- 2. Feature Engineering ---")
    fe = FeatureEngineer()
    df = fe.extract_temporal_features(df)
    df = fe.compute_historical_density(df)
    df = fe.compute_corridor_saturation(df)
    df = fe.compute_station_resolution_speed(df)
    df = fe.encode_categoricals(df, is_train=True) # fits TargetEncoder
    df = fe.add_distance_features(df)
    features_list = fe.get_final_feature_columns()
    fe.save(df, 'data/processed/features.csv')
    
    # 3. Train/test split for all 3 targets
    print("\n--- 3. Train/Test Splits ---")
    severity_model = SeverityClassifier()
    X_train_sev, X_test_sev, y_train_sev, y_test_sev = severity_model.prepare_data(df)
    
    duration_model = DurationRegressor()
    X_train_dur, X_test_dur, y_train_dur, y_test_dur, test_causes_dur = duration_model.prepare_data(df)
    
    closure_model = ClosureClassifier()
    X_train_clo, X_test_clo, y_train_clo, y_test_clo = closure_model.prepare_data(df)
    
    if not args.skip_train:
        # 4. Train severity model -> save
        print("\n--- 4. Train Severity Model ---")
        severity_model.train(X_train_sev, y_train_sev)
        severity_model.save()
        
        # 5. Train duration model -> save
        print("\n--- 5. Train Duration Model ---")
        duration_model.train(X_train_dur, y_train_dur)
        duration_model.save()
        
        # 6. Train closure model -> save
        print("\n--- 6. Train Closure Model ---")
        closure_model.train(X_train_clo, y_train_clo)
        closure_model.save()
    else:
        print("\n--- Skipping Training (Loading Models) ---")
        severity_model.load()
        duration_model.load()
        closure_model.load()
        
    # 7. Evaluate all models
    print("\n--- 7. Evaluation ---")
    severity_model.evaluate(X_test_sev, y_test_sev)
    duration_model.evaluate(X_test_dur, y_test_dur, test_causes_dur)
    closure_model.evaluate(X_test_clo, y_test_clo)
    
    # 8. Build resource lookup table
    print("\n--- 8. Recommender Lookup Table ---")
    recommender = ResourceRecommender()
    recommender.build_lookup_table(df)
    
    # 9. Pipeline complete
    print("\nPipeline complete. Run: streamlit run dashboard/app.py")

if __name__ == "__main__":
    main()
