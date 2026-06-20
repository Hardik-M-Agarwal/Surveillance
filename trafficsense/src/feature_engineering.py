import pandas as pd
import numpy as np
import json
import os
import joblib
from category_encoders import TargetEncoder
from haversine import haversine, Unit

class FeatureEngineer:
    def __init__(self):
        self.target_encoder = None
        self.top_15_junctions = [
            'MekhriCircle', 'AyyappaTempleJunc', 'SatteliteBusStandJunc', 
            'YeshwanthpuraCircle', 'YelhankaCircle', 'SilkBoardJunc',
            'HebbalJunc', 'KRPuramJunc', 'TinFactoryJunc', 'GoraguntepalyaJunc',
            'SarakkiJunc', 'KadubeesanahalliJunc', 'MarathahalliJunc', 'DairyCircle', 'NayandahalliJunc'
        ] # Added some reasonable fallbacks to make 15, based on prompt info top ones are listed

    def extract_temporal_features(self, df):
        print("Extracting temporal features...")
        df['hour'] = df['start_datetime'].dt.hour
        
        def get_hour_bin(h):
            if 0 <= h <= 3: return 'late_night'
            elif 4 <= h <= 6: return 'early_morning'
            elif 7 <= h <= 10: return 'morning'
            elif 11 <= h <= 15: return 'midday'
            elif 16 <= h <= 19: return 'evening'
            else: return 'night'
            
        df['hour_bin'] = df['hour'].apply(get_hour_bin)
        df['day_of_week'] = df['start_datetime'].dt.dayofweek
        df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)
        df['month'] = df['start_datetime'].dt.month
        
        peak_hours = [19, 20, 21, 22, 4, 5, 6]
        df['is_peak_hour'] = df['hour'].isin(peak_hours).astype(int)
        
        return df

    def compute_historical_density(self, df, window_days=30):
        print("Computing historical density...")
        
        # Keep original index to restore order later
        df['orig_index'] = df.index
        
        # Sort and filter NaT for rolling calculation
        valid_df = df.dropna(subset=['start_datetime']).sort_values('start_datetime').copy()
        invalid_df = df[df['start_datetime'].isna()].copy()
        
        valid_df['dummy'] = 1
        
        counts = []
        for name, group in valid_df.groupby('corridor'):
            # Set index for rolling
            group_indexed = group.set_index('start_datetime')
            r_count = group_indexed['dummy'].rolling(f'{window_days}D').sum().values - 1
            group['historical_density_30d'] = r_count
            counts.append(group)
            
        if len(counts) > 0:
            valid_df = pd.concat(counts)
            valid_df.drop(columns=['dummy'], inplace=True)
            
        invalid_df['historical_density_30d'] = 0.0
        
        df = pd.concat([valid_df, invalid_df]).sort_values('orig_index')
        df.drop(columns=['orig_index'], inplace=True)
        return df.reset_index(drop=True)

    def compute_corridor_saturation(self, df):
        print("Computing corridor saturation...")
        # For each incident, count how many OTHER incidents with status = 'active' existed on the same corridor
        # Since 'status' is a static snapshot in this dataset, this feature is an approximation.
        # We'll count incidents that started before this one and closed after this one.
        # If closed_datetime is NaT, it's considered active indefinitely.
        
        df = df.sort_values('start_datetime')
        
        # Very slow if O(N^2), but N is 8173, so O(N^2) is ~66M operations, which is fast enough in vectorized numpy/pandas.
        start_vals = df['start_datetime'].values
        close_vals = df['closed_datetime'].fillna(pd.Timestamp.max.tz_localize('UTC')).values
        corridor_vals = df['corridor'].values
        
        saturation = np.zeros(len(df), dtype=int)
        for i in range(len(df)):
            t = start_vals[i]
            c = corridor_vals[i]
            # active when: start < t and close > t and same corridor
            active_mask = (start_vals < t) & (close_vals > t) & (corridor_vals == c)
            saturation[i] = active_mask.sum()
            
        df['corridor_active_count'] = saturation
        return df

    def compute_station_resolution_speed(self, df):
        print("Computing station resolution speed...")
        # Exponential moving average of duration_minutes across historical records
        # Per (police_station, event_cause) pair
        df = df.sort_values('start_datetime')
        
        global_median = df['duration_minutes'].median()
        
        emas = []
        for name, group in df.groupby(['police_station', 'event_cause']):
            group = group.sort_values('start_datetime')
            # fillna to avoid EMA dropping NaN rows, but duration_minutes can be NaN
            # We compute EMA on known durations, then forward fill/shift
            known = group['duration_minutes'].dropna()
            if len(known) > 0:
                ema = known.ewm(span=10, adjust=False).mean()
                # We need the EMA *before* the current incident
                ema_shifted = ema.shift(1)
                group['station_cause_avg_duration'] = ema_shifted
                # Map back to group
                group['station_cause_avg_duration'] = group['station_cause_avg_duration'].ffill()
            else:
                group['station_cause_avg_duration'] = np.nan
            emas.append(group)
            
        df = pd.concat(emas).sort_values('start_datetime').reset_index(drop=True)
        df['station_cause_avg_duration'] = df['station_cause_avg_duration'].fillna(global_median)
        return df

    def encode_categoricals(self, df, is_train=True):
        print(f"Encoding categoricals (is_train={is_train})...")
        
        # Target encode: event_cause, corridor, police_station
        cols_to_encode = ['event_cause', 'corridor', 'police_station']
        
        if is_train:
            self.target_encoder = TargetEncoder(cols=cols_to_encode)
            encoded = self.target_encoder.fit_transform(df[cols_to_encode], df['requires_road_closure'].astype(int))
            # Save the encoder
            os.makedirs('models', exist_ok=True)
            joblib.dump(self.target_encoder, 'models/label_encoders.pkl')
        else:
            if self.target_encoder is None:
                if os.path.exists('models/label_encoders.pkl'):
                    self.target_encoder = joblib.load('models/label_encoders.pkl')
                else:
                    raise ValueError("TargetEncoder not fit yet and no saved model found.")
            encoded = self.target_encoder.transform(df[cols_to_encode])
            
        df['event_cause_encoded'] = encoded['event_cause']
        df['corridor_encoded'] = encoded['corridor']
        df['police_station_encoded'] = encoded['police_station']
        
        # Binary encode
        df['priority_encoded'] = (df['priority'] == 'High').astype(int)
        df['event_type_encoded'] = (df['event_type'] == 'planned').astype(int)
        if 'requires_road_closure' in df.columns and df['requires_road_closure'].dtype == bool:
            df['requires_road_closure'] = df['requires_road_closure'].astype(int)
            
        # One-hot encode veh_type
        veh_types = ['bmtc_bus', 'heavy_vehicle', 'lcv', 'others', 'private_bus', 
                     'private_car', 'truck', 'ksrtc_bus', 'taxi', 'auto', 'unknown']
        
        # Create these columns safely
        for vt in veh_types:
            df[f'veh_type_{vt}'] = (df['veh_type'] == vt).astype(int)
            
        return df

    def add_distance_features(self, df):
        print("Adding distance features...")
        def calc_displacement(row):
            if pd.isna(row['latitude']) or pd.isna(row['longitude']) or \
               pd.isna(row['resolved_at_latitude']) or pd.isna(row['resolved_at_longitude']):
                return 0.0
            return haversine((row['latitude'], row['longitude']), 
                             (row['resolved_at_latitude'], row['resolved_at_longitude']), unit=Unit.KILOMETERS)
                             
        df['displacement_km'] = df.apply(calc_displacement, axis=1)
        
        df['is_junction_hotspot'] = df['junction'].isin(self.top_15_junctions).astype(int)
        return df

    def get_final_feature_columns(self):
        FEATURE_COLUMNS = [
            'event_type_encoded',       # planned=1, unplanned=0
            'event_cause_encoded',      # target encoded float
            'corridor_encoded',         # target encoded float
            'police_station_encoded',   # target encoded float
            'hour',                     # 0-23 int
            'day_of_week',              # 0-6 int
            'is_weekend',               # binary
            'is_peak_hour',             # binary
            'month',                    # 1-12 int
            'requires_road_closure',    # binary (only use for duration + closure models, NOT for closure model as target)
            'priority_encoded',         # High=1, Low=0
            'historical_density_30d',   # count float
            'corridor_active_count',    # count int
            'station_cause_avg_duration',# float minutes
            'is_junction_hotspot',      # binary
            'displacement_km',          # float, 0 if unavailable
            # one-hot veh_type columns:
            'veh_type_bmtc_bus', 'veh_type_heavy_vehicle', 'veh_type_lcv',
            'veh_type_others', 'veh_type_private_bus', 'veh_type_private_car',
            'veh_type_truck', 'veh_type_ksrtc_bus', 'veh_type_taxi',
            'veh_type_auto', 'veh_type_unknown'
        ]
        
        os.makedirs('models', exist_ok=True)
        with open('models/feature_columns.json', 'w') as f:
            json.dump(FEATURE_COLUMNS, f)
            
        return FEATURE_COLUMNS

    def save(self, df, output_path):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f"Saving feature engineered data to {output_path}...")
        df.to_csv(output_path, index=False)
        return df
