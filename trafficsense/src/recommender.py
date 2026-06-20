import pandas as pd
import os

class ResourceRecommender:
    def __init__(self):
        self.lookup_table = None

    def build_lookup_table(self, df):
        print("Building resource lookup table...")
        # from cleaned dataset
        
        # Proxy for median_constables: High+Critical=5, Medium=2.5, Low=1.5
        # Add 2 if requires_road_closure=True
        
        def compute_constables(row):
            base = 1.5
            if row['priority'] == 'High':
                base = 5.0 # Covers High+Critical and High
            
            # Simple heuristic
            if row['severity'] in ['Critical', 'High']:
                base = 5
            elif row['severity'] == 'Medium':
                base = 3
            else:
                base = 2
                
            if row['requires_road_closure']:
                base += 2
            return base

        df['constable_proxy'] = df.apply(compute_constables, axis=1)
        
        # Group by (event_cause, severity, corridor)
        lookup = df.groupby(['event_cause', 'severity', 'corridor']).agg(
            median_constables=('constable_proxy', 'median'),
            road_closure_typical=('requires_road_closure', lambda x: x.mode()[0] if len(x.mode())>0 else False),
            typical_duration_min=('duration_minutes', 'median'),
            suggested_police_station=('police_station', lambda x: x.mode()[0] if len(x.mode())>0 else 'unknown')
        ).reset_index()
        
        os.makedirs('data/processed', exist_ok=True)
        lookup.to_csv('data/processed/resource_lookup.csv', index=False)
        self.lookup_table = lookup
        return lookup

    def load_lookup_table(self, path='data/processed/resource_lookup.csv'):
        if os.path.exists(path):
            self.lookup_table = pd.read_csv(path)
        else:
            print(f"Warning: {path} not found.")

    def recommend(self, event_cause, predicted_severity, corridor, hour, requires_closure_prob, estimated_duration=None):
        if self.lookup_table is None:
            self.load_lookup_table()
            
        # Match
        match = self.lookup_table[
            (self.lookup_table['event_cause'] == event_cause) &
            (self.lookup_table['severity'] == predicted_severity) &
            (self.lookup_table['corridor'] == corridor)
        ]
        
        confidence = "high"
        if len(match) > 0:
            row = match.iloc[0]
        else:
            # Fallback to event_cause + predicted_severity
            match_fallback = self.lookup_table[
                (self.lookup_table['event_cause'] == event_cause) &
                (self.lookup_table['severity'] == predicted_severity)
            ]
            confidence = "medium"
            if len(match_fallback) > 0:
                row = match_fallback.mean(numeric_only=True)
                # Need mode for string column
                row['suggested_police_station'] = match_fallback['suggested_police_station'].mode()[0]
            else:
                confidence = "low"
                # Hard fallback
                row = pd.Series({
                    'median_constables': 3,
                    'suggested_police_station': 'Central Station'
                })
                
        base_constables = int(row.get('median_constables', 3))
        
        # Peak hours: [19, 20, 21, 22, 4, 5, 6]
        peak_hours = [19, 20, 21, 22, 4, 5, 6]
        if hour in peak_hours:
            base_constables += 2
            
        if requires_closure_prob > 0.6:
            base_constables += 3
            
        barricades = base_constables * 2
        diversion = True if requires_closure_prob > 0.5 else False
        
        # Alert level logic
        if predicted_severity == 'Critical' or requires_closure_prob > 0.8:
            alert = 'CRITICAL'
        elif predicted_severity == 'High' or requires_closure_prob > 0.6:
            alert = 'RED'
        elif predicted_severity == 'Medium' or requires_closure_prob > 0.4:
            alert = 'AMBER'
        else:
            alert = 'GREEN'
            
        clearance = estimated_duration if estimated_duration is not None else row.get('typical_duration_min', 60)
        if pd.isna(clearance): clearance = 60
            
        return {
            'constable_count': int(base_constables),
            'barricade_count': int(barricades),
            'diversion_needed': bool(diversion),
            'suggested_police_station': str(row.get('suggested_police_station', 'Unknown')),
            'estimated_clearance_minutes': float(clearance),
            'confidence': confidence,
            'alert_level': alert
        }
