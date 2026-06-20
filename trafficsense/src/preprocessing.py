import pandas as pd
import numpy as np
from geopy.geocoders import Nominatim
from geopy.exc import GeocoderTimedOut, GeocoderUnavailable

class DataCleaner:
    def __init__(self):
        self.df = None

    def load(self, filepath):
        print(f"Loading data from {filepath}...")
        self.df = pd.read_csv(filepath)
        
        datetime_cols = ['start_datetime', 'closed_datetime', 'modified_datetime', 'created_date', 'resolved_datetime']
        for col in datetime_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_datetime(self.df[col], utc=True, errors='coerce')
        return self.df

    def drop_useless_columns(self):
        print("Dropping useless columns...")
        cols_to_drop = [
            'map_file', 'comment', 'meta_data', 'direction', 
            'route_path', 'assigned_to_police_id', 'citizen_accident_id'
        ]
        self.df = self.df.drop(columns=[col for col in cols_to_drop if col in self.df.columns])
        return self.df

    def clean_event_cause(self):
        print("Cleaning event causes...")
        if 'event_cause' in self.df.columns:
            # Merge duplicate debris
            self.df['event_cause'] = self.df['event_cause'].replace({'Debris': 'debris'})
            # Merge Fog
            self.df['event_cause'] = self.df['event_cause'].replace({'Fog / Low Visibility': 'fog_low_visibility'})
            
            # Drop test_demo
            self.df = self.df[self.df['event_cause'] != 'test_demo'].copy()
            
        # Strip whitespace from string columns
        string_cols = self.df.select_dtypes(include=['object', 'string']).columns
        for col in string_cols:
            self.df[col] = self.df[col].apply(lambda x: x.strip() if isinstance(x, str) else x)
            
        return self.df

    def compute_duration(self):
        print("Computing duration...")
        if 'start_datetime' in self.df.columns and 'closed_datetime' in self.df.columns:
            # Calculate duration in minutes
            self.df['duration_minutes'] = (self.df['closed_datetime'] - self.df['start_datetime']).dt.total_seconds() / 60
            
            # Drop negative durations (clock errors)
            self.df = self.df[(self.df['duration_minutes'] >= 0) | (self.df['duration_minutes'].isna())].copy()
            
            # Infrastructure causes
            infra_causes = ['pot_holes', 'road_conditions', 'debris', 'construction']
            self.df['is_infrastructure'] = self.df['event_cause'].isin(infra_causes)
            
            # Cap infrastructure causes at 72 hours (4320 min)
            infra_mask = self.df['is_infrastructure'] & (self.df['duration_minutes'] > 4320)
            self.df.loc[infra_mask, 'duration_minutes'] = 4320
            
            # Note: For closed_datetime == NaN, duration_minutes will naturally be NaN, which is correct
            
        return self.df

    def assign_severity_label(self):
        print("Assigning severity labels...")
        def assign_severity(row):
            duration = row['duration_minutes'] if not pd.isna(row['duration_minutes']) else 0
            if row['requires_road_closure'] and row['priority'] == 'High' and duration > 120:
                return 'Critical'
            elif row['priority'] == 'High' and row['requires_road_closure']:
                return 'High'
            elif row['priority'] == 'High':
                return 'Medium'
            else:
                return 'Low'
                
        self.df['severity'] = self.df.apply(assign_severity, axis=1)
        return self.df

    def clean_corridors(self):
        print("Cleaning corridors...")
        if 'corridor' in self.df.columns:
            # Strip whitespace
            self.df['corridor'] = self.df['corridor'].apply(lambda x: x.strip() if isinstance(x, str) else x)
            
            # Fill NaN
            nan_mask = self.df['corridor'].isna()
            if nan_mask.sum() > 0:
                print(f"Filling {nan_mask.sum()} NaN corridors...")
                # Fallback directly to "Non-corridor" as per specs
                self.df.loc[nan_mask, 'corridor'] = "Non-corridor"
                
        return self.df

    def clean_vehicle_type(self):
        print("Cleaning vehicle types...")
        if 'veh_type' in self.df.columns and 'event_cause' in self.df.columns:
            non_vehicle_events = ['tree_fall', 'protest', 'public_event', 'pot_holes', 'water_logging']
            mask = self.df['event_cause'].isin(non_vehicle_events) & self.df['veh_type'].isna()
            self.df.loc[mask, 'veh_type'] = 'unknown'
            
        return self.df

    def save(self, output_path):
        import os
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        print(f"Saving cleaned data to {output_path}...")
        self.df.to_csv(output_path, index=False)
        return self.df

    def run_pipeline(self, input_path, output_path):
        self.load(input_path)
        self.drop_useless_columns()
        self.clean_event_cause()
        self.compute_duration()
        self.assign_severity_label()
        self.clean_corridors()
        self.clean_vehicle_type()
        self.save(output_path)
        return self.df
