"""
Advanced Data Science Analytics Module for Medicine Inventory Management
"""

import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from typing import List, Dict, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
import joblib
import json
from sqlalchemy import func, and_, or_
from .. import db
from ..models import Medicine, ConsumptionRecord, AnalyticsCache, InventoryAlert, User


class InventoryAnalytics:
    """Advanced analytics for inventory management using data science techniques"""
    
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.scaler = StandardScaler()
        self.demand_model = None
        self.consumption_model = None
    
    def get_consumption_data(self, days: int = 90) -> pd.DataFrame:
        """Get consumption data for analysis"""
        start_date = date.today() - timedelta(days=days)
        
        query = db.session.query(
            ConsumptionRecord.consumption_date,
            ConsumptionRecord.quantity_consumed,
            Medicine.name,
            Medicine.category,
            Medicine.id.label('medicine_id')
        ).join(Medicine).filter(
            ConsumptionRecord.user_id == self.user_id,
            ConsumptionRecord.consumption_date >= start_date
        )
        
        df = pd.read_sql(query.statement, db.engine)
        return df
    
    def calculate_consumption_rates(self) -> Dict[int, float]:
        """Calculate consumption rates for each medicine"""
        consumption_data = self.get_consumption_data()
        
        if consumption_data.empty:
            return {}
        
        # Group by medicine and calculate daily consumption rate
        daily_rates = consumption_data.groupby('medicine_id').agg({
            'quantity_consumed': 'sum',
            'consumption_date': 'nunique'
        }).reset_index()
        
        daily_rates['consumption_rate'] = daily_rates['quantity_consumed'] / daily_rates['consumption_date']
        
        return dict(zip(daily_rates['medicine_id'], daily_rates['consumption_rate']))
    
    def forecast_demand(self, medicine_id: int, days_ahead: int = 30) -> float:
        """Forecast demand for a specific medicine using time series analysis"""
        consumption_data = self.get_consumption_data(180)  # Use 6 months of data
        
        if consumption_data.empty:
            return 0.0
        
        # Filter for specific medicine
        med_data = consumption_data[consumption_data['medicine_id'] == medicine_id]
        
        if len(med_data) < 7:  # Need at least a week of data
            return 0.0
        
        # Create time series
        med_data['consumption_date'] = pd.to_datetime(med_data['consumption_date'])
        daily_consumption = med_data.groupby('consumption_date')['quantity_consumed'].sum().resample('D').sum().fillna(0)
        
        # Simple moving average forecast
        window_size = min(14, len(daily_consumption) // 2)
        if window_size < 3:
            return daily_consumption.mean()
        
        forecast = daily_consumption.rolling(window=window_size).mean().iloc[-1]
        return max(0, forecast * days_ahead)
    
    def calculate_abc_analysis(self) -> Dict[str, List[Dict]]:
        """Perform ABC analysis based on consumption value"""
        medicines = Medicine.query.filter(Medicine.user_id == self.user_id).all()
        consumption_rates = self.calculate_consumption_rates()
        
        analysis_data = []
        for med in medicines:
            consumption_rate = consumption_rates.get(med.id, 0)
            value = consumption_rate * (med.selling_price or 0)
            analysis_data.append({
                'medicine_id': med.id,
                'name': med.name,
                'consumption_rate': consumption_rate,
                'value': value,
                'quantity': med.quantity
            })
        
        # Sort by value (descending)
        analysis_data.sort(key=lambda x: x['value'], reverse=True)
        
        total_value = sum(item['value'] for item in analysis_data)
        cumulative_value = 0
        
        abc_categories = {'A': [], 'B': [], 'C': []}
        
        for item in analysis_data:
            cumulative_value += item['value']
            percentage = (cumulative_value / total_value) * 100 if total_value > 0 else 0
            
            if percentage <= 80:
                category = 'A'
            elif percentage <= 95:
                category = 'B'
            else:
                category = 'C'
            
            item['category'] = category
            item['cumulative_percentage'] = percentage
            abc_categories[category].append(item)
        
        return abc_categories
    
    def calculate_xyz_analysis(self) -> Dict[str, List[Dict]]:
        """Perform XYZ analysis based on demand variability"""
        consumption_data = self.get_consumption_data(90)
        
        if consumption_data.empty:
            return {'X': [], 'Y': [], 'Z': []}
        
        xyz_categories = {'X': [], 'Y': [], 'Z': []}
        
        for medicine_id in consumption_data['medicine_id'].unique():
            med_data = consumption_data[consumption_data['medicine_id'] == medicine_id]
            daily_consumption = med_data.groupby('consumption_date')['quantity_consumed'].sum()
            
            if len(daily_consumption) < 7:
                continue
            
            # Calculate coefficient of variation
            mean_consumption = daily_consumption.mean()
            std_consumption = daily_consumption.std()
            cv = (std_consumption / mean_consumption) * 100 if mean_consumption > 0 else 0
            
            medicine = Medicine.query.get(medicine_id)
            if not medicine:
                continue
            
            item = {
                'medicine_id': medicine_id,
                'name': medicine.name,
                'mean_consumption': mean_consumption,
                'std_consumption': std_consumption,
                'coefficient_of_variation': cv
            }
            
            if cv <= 25:
                category = 'X'  # Low variability
            elif cv <= 50:
                category = 'Y'  # Medium variability
            else:
                category = 'Z'  # High variability
            
            item['category'] = category
            xyz_categories[category].append(item)
        
        return xyz_categories
    
    def generate_inventory_insights(self) -> Dict:
        """Generate comprehensive inventory insights"""
        medicines = Medicine.query.filter(Medicine.user_id == self.user_id).all()
        
        if not medicines:
            return {}
        
        # Basic statistics
        total_medicines = len(medicines)
        expired_medicines = [m for m in medicines if m.is_expired()]
        expiring_soon = [m for m in medicines if 0 < m.days_until_expiry() <= 30]
        low_stock = [m for m in medicines if m.is_low_stock()]
        overstocked = [m for m in medicines if m.is_overstocked()]
        
        # Calculate total inventory value
        total_value = sum(m.quantity * (m.selling_price or 0) for m in medicines)
        
        # Category distribution
        categories = {}
        for med in medicines:
            cat = med.category or 'Uncategorized'
            categories[cat] = categories.get(cat, 0) + 1
        
        # Risk analysis
        high_risk = [m for m in medicines if m.calculate_risk_score() > 0.7]
        medium_risk = [m for m in medicines if 0.4 < m.calculate_risk_score() <= 0.7]
        low_risk = [m for m in medicines if m.calculate_risk_score() <= 0.4]
        
        return {
            'total_medicines': total_medicines,
            'expired_count': len(expired_medicines),
            'expiring_soon_count': len(expiring_soon),
            'low_stock_count': len(low_stock),
            'overstocked_count': len(overstocked),
            'total_inventory_value': total_value,
            'categories': categories,
            'risk_distribution': {
                'high_risk': len(high_risk),
                'medium_risk': len(medium_risk),
                'low_risk': len(low_risk)
            },
            'abc_analysis': self.calculate_abc_analysis(),
            'xyz_analysis': self.calculate_xyz_analysis()
        }
    
    def generate_recommendations(self) -> List[Dict]:
        """Generate actionable recommendations based on analytics"""
        recommendations = []
        medicines = Medicine.query.filter(Medicine.user_id == self.user_id).all()
        
        for med in medicines:
            # Expiry recommendations
            if med.is_expired():
                recommendations.append({
                    'type': 'critical',
                    'medicine_id': med.id,
                    'medicine_name': med.name,
                    'message': f'{med.name} has expired and should be removed',
                    'action': 'remove_expired'
                })
            elif med.days_until_expiry() <= 7:
                recommendations.append({
                    'type': 'high',
                    'medicine_id': med.id,
                    'medicine_name': med.name,
                    'message': f'{med.name} expires in {med.days_until_expiry()} days',
                    'action': 'use_soon'
                })
            
            # Stock level recommendations
            if med.is_low_stock():
                recommendations.append({
                    'type': 'medium',
                    'medicine_id': med.id,
                    'medicine_name': med.name,
                    'message': f'{med.name} is running low (quantity: {med.quantity})',
                    'action': 'reorder'
                })
            
            if med.is_overstocked():
                recommendations.append({
                    'type': 'low',
                    'medicine_id': med.id,
                    'medicine_name': med.name,
                    'message': f'{med.name} is overstocked (quantity: {med.quantity})',
                    'action': 'reduce_stock'
                })
            
            # Risk-based recommendations
            risk_score = med.calculate_risk_score()
            if risk_score > 0.8:
                recommendations.append({
                    'type': 'high',
                    'medicine_id': med.id,
                    'medicine_name': med.name,
                    'message': f'{med.name} has high risk score ({risk_score:.2f})',
                    'action': 'review_urgently'
                })
        
        return sorted(recommendations, key=lambda x: ['critical', 'high', 'medium', 'low'].index(x['type']))
    
    def create_consumption_record(self, medicine_id: int, quantity: int, notes: str = None) -> ConsumptionRecord:
        """Create a new consumption record"""
        record = ConsumptionRecord(
            medicine_id=medicine_id,
            user_id=self.user_id,
            quantity_consumed=quantity,
            consumption_date=date.today(),
            notes=notes
        )
        db.session.add(record)
        
        # Update medicine quantity
        medicine = Medicine.query.get(medicine_id)
        if medicine and medicine.user_id == self.user_id:
            medicine.quantity = max(0, medicine.quantity - quantity)
            medicine.last_consumption_date = date.today()
            
            # Update consumption rate
            consumption_rates = self.calculate_consumption_rates()
            medicine.consumption_rate = consumption_rates.get(medicine_id, 0)
            
            # Update risk score
            medicine.risk_score = medicine.calculate_risk_score()
        
        db.session.commit()
        return record
    
    def get_cached_analytics(self, cache_key: str) -> Optional[Dict]:
        """Get cached analytics data"""
        cache = AnalyticsCache.query.filter(
            AnalyticsCache.user_id == self.user_id,
            AnalyticsCache.cache_key == cache_key,
            AnalyticsCache.expires_at > datetime.utcnow()
        ).first()
        
        return cache.get_data() if cache else None
    
    def cache_analytics(self, cache_key: str, data: Dict, hours: int = 24) -> None:
        """Cache analytics data"""
        # Remove old cache
        AnalyticsCache.query.filter(
            AnalyticsCache.user_id == self.user_id,
            AnalyticsCache.cache_key == cache_key
        ).delete()
        
        # Create new cache
        cache = AnalyticsCache(
            user_id=self.user_id,
            cache_key=cache_key,
            cache_data=json.dumps(data),
            expires_at=datetime.utcnow() + timedelta(hours=hours)
        )
        db.session.add(cache)
        db.session.commit()


class PredictiveAnalytics:
    """Machine Learning based predictive analytics"""
    
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.models = {}
    
    def prepare_training_data(self) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data for demand forecasting"""
        # Get historical consumption data
        consumption_data = self.get_consumption_data(365)  # 1 year of data
        
        if consumption_data.empty:
            return pd.DataFrame(), pd.Series()
        
        # Create features
        features = []
        targets = []
        
        for medicine_id in consumption_data['medicine_id'].unique():
            med_data = consumption_data[consumption_data['medicine_id'] == medicine_id]
            daily_consumption = med_data.groupby('consumption_date')['quantity_consumed'].sum().resample('D').sum().fillna(0)
            
            if len(daily_consumption) < 30:  # Need sufficient data
                continue
            
            # Create time-based features
            for i in range(7, len(daily_consumption)):
                feature_vector = [
                    daily_consumption.iloc[i-7:i].mean(),  # 7-day average
                    daily_consumption.iloc[i-7:i].std(),   # 7-day std
                    daily_consumption.iloc[i-14:i-7].mean(),  # Previous week average
                    daily_consumption.iloc[i-30:i].mean(),  # 30-day average
                    i % 7,  # Day of week
                    i % 30,  # Day of month
                ]
                
                features.append(feature_vector)
                targets.append(daily_consumption.iloc[i])
        
        return pd.DataFrame(features), pd.Series(targets)
    
    def train_demand_model(self) -> None:
        """Train demand forecasting model"""
        X, y = self.prepare_training_data()
        
        if X.empty or len(y) < 50:  # Need sufficient data
            return
        
        # Train Random Forest model
        self.demand_model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.demand_model.fit(X, y)
        
        # Save model
        joblib.dump(self.demand_model, f'models/demand_model_{self.user_id}.pkl')
    
    def predict_demand(self, medicine_id: int, days_ahead: int = 7) -> List[float]:
        """Predict demand for next N days"""
        if not self.demand_model:
            self.load_demand_model()
        
        if not self.demand_model:
            return [0.0] * days_ahead
        
        # Get recent consumption data for the medicine
        consumption_data = self.get_consumption_data(30)
        med_data = consumption_data[consumption_data['medicine_id'] == medicine_id]
        
        if med_data.empty:
            return [0.0] * days_ahead
        
        daily_consumption = med_data.groupby('consumption_date')['quantity_consumed'].sum().resample('D').sum().fillna(0)
        
        if len(daily_consumption) < 7:
            return [0.0] * days_ahead
        
        predictions = []
        current_data = daily_consumption.iloc[-7:].values
        
        for day in range(days_ahead):
            # Create feature vector
            feature_vector = [
                current_data[-7:].mean(),  # 7-day average
                current_data[-7:].std(),   # 7-day std
                current_data[-14:-7].mean() if len(current_data) >= 14 else current_data[-7:].mean(),  # Previous week
                current_data.mean(),  # Overall average
                (len(current_data) + day) % 7,  # Day of week
                (len(current_data) + day) % 30,  # Day of month
            ]
            
            prediction = self.demand_model.predict([feature_vector])[0]
            predictions.append(max(0, prediction))
            
            # Update current_data for next prediction
            current_data = np.append(current_data[1:], prediction)
        
        return predictions
    
    def load_demand_model(self) -> None:
        """Load trained demand model"""
        try:
            self.demand_model = joblib.load(f'models/demand_model_{self.user_id}.pkl')
        except FileNotFoundError:
            self.demand_model = None
    
    def get_consumption_data(self, days: int = 90) -> pd.DataFrame:
        """Get consumption data for analysis"""
        start_date = date.today() - timedelta(days=days)
        
        query = db.session.query(
            ConsumptionRecord.consumption_date,
            ConsumptionRecord.quantity_consumed,
            Medicine.id.label('medicine_id')
        ).join(Medicine).filter(
            ConsumptionRecord.user_id == self.user_id,
            ConsumptionRecord.consumption_date >= start_date
        )
        
        return pd.read_sql(query.statement, db.engine)
