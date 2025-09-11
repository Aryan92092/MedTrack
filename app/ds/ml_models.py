"""
Machine Learning Models for Medicine Inventory Management
"""

import pandas as pd
import numpy as np
from datetime import date, datetime, timedelta
from typing import List, Dict, Tuple, Optional
from sklearn.ensemble import RandomForestRegressor, IsolationForest
from sklearn.linear_model import LinearRegression
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.cluster import KMeans
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
import joblib
import json
from sqlalchemy import func, and_, or_
from .. import db
from ..models import Medicine, ConsumptionRecord, User


class DemandForecastingModel:
    """Machine Learning model for demand forecasting"""
    
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.model = RandomForestRegressor(n_estimators=100, random_state=42)
        self.scaler = StandardScaler()
        self.label_encoders = {}
        self.is_trained = False
        self.feature_columns = [
            'quantity', 'days_until_expiry', 'consumption_rate', 'seasonality',
            'day_of_week', 'month', 'category_encoded', 'price_tier'
        ]
    
    def prepare_training_data(self, days: int = 365) -> Tuple[pd.DataFrame, pd.Series]:
        """Prepare training data for demand forecasting"""
        # Get consumption data
        start_date = date.today() - timedelta(days=days)
        
        query = db.session.query(
            ConsumptionRecord.consumption_date,
            ConsumptionRecord.quantity_consumed,
            Medicine.id.label('medicine_id'),
            Medicine.name,
            Medicine.category,
            Medicine.quantity,
            Medicine.expiry_date,
            Medicine.selling_price
        ).join(Medicine).filter(
            ConsumptionRecord.user_id == self.user_id,
            ConsumptionRecord.consumption_date >= start_date
        )
        
        df = pd.read_sql(query.statement, db.engine)
        
        if df.empty:
            return pd.DataFrame(), pd.Series()
        
        # Create features
        features = []
        targets = []
        
        for medicine_id in df['medicine_id'].unique():
            med_data = df[df['medicine_id'] == medicine_id].copy()
            med_info = med_data.iloc[0]
            
            # Create time series features
            med_data['consumption_date'] = pd.to_datetime(med_data['consumption_date'])
            daily_consumption = med_data.groupby('consumption_date')['quantity_consumed'].sum().resample('D').sum().fillna(0)
            
            if len(daily_consumption) < 30:  # Need sufficient data
                continue
            
            # Calculate consumption rate
            consumption_rate = daily_consumption.mean()
            
            # Create features for each day
            for i in range(7, len(daily_consumption)):
                current_date = daily_consumption.index[i]
                
                # Time-based features
                day_of_week = current_date.weekday()
                month = current_date.month
                seasonality = self._calculate_seasonality(current_date)
                
                # Medicine-specific features
                days_until_expiry = (med_info['expiry_date'] - current_date.date()).days
                price_tier = self._calculate_price_tier(med_info['selling_price'])
                
                # Category encoding
                category_encoded = self._encode_category(med_info['category'])
                
                # Historical consumption features
                recent_consumption = daily_consumption.iloc[i-7:i].mean()
                weekly_consumption = daily_consumption.iloc[i-14:i-7].mean()
                
                feature_vector = [
                    med_info['quantity'],
                    days_until_expiry,
                    consumption_rate,
                    seasonality,
                    day_of_week,
                    month,
                    category_encoded,
                    price_tier,
                    recent_consumption,
                    weekly_consumption
                ]
                
                features.append(feature_vector)
                targets.append(daily_consumption.iloc[i])
        
        if not features:
            return pd.DataFrame(), pd.Series()
        
        feature_df = pd.DataFrame(features, columns=self.feature_columns + ['recent_consumption', 'weekly_consumption'])
        target_series = pd.Series(targets)
        
        return feature_df, target_series
    
    def train(self) -> Dict[str, float]:
        """Train the demand forecasting model"""
        X, y = self.prepare_training_data()
        
        if X.empty or len(y) < 50:
            return {'error': 'Insufficient training data'}
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
        
        # Scale features
        X_train_scaled = self.scaler.fit_transform(X_train)
        X_test_scaled = self.scaler.transform(X_test)
        
        # Train model
        self.model.fit(X_train_scaled, y_train)
        
        # Make predictions
        y_pred = self.model.predict(X_test_scaled)
        
        # Calculate metrics
        mae = mean_absolute_error(y_test, y_pred)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        
        self.is_trained = True
        
        # Save model
        self.save_model()
        
        return {
            'mae': mae,
            'mse': mse,
            'rmse': np.sqrt(mse),
            'r2': r2,
            'training_samples': len(X_train),
            'test_samples': len(X_test)
        }
    
    def predict_demand(self, medicine_id: int, days_ahead: int = 7) -> List[float]:
        """Predict demand for a specific medicine"""
        if not self.is_trained:
            self.load_model()
        
        if not self.is_trained:
            return [0.0] * days_ahead
        
        # Get medicine information
        medicine = Medicine.query.get(medicine_id)
        if not medicine or medicine.user_id != self.user_id:
            return [0.0] * days_ahead
        
        # Get recent consumption data
        consumption_data = self._get_recent_consumption(medicine_id, 30)
        
        if consumption_data.empty:
            return [0.0] * days_ahead
        
        predictions = []
        current_date = date.today()
        
        for day in range(days_ahead):
            target_date = current_date + timedelta(days=day)
            
            # Create feature vector
            feature_vector = self._create_feature_vector(medicine, target_date, consumption_data)
            
            if feature_vector is None:
                predictions.append(0.0)
                continue
            
            # Scale features
            feature_vector_scaled = self.scaler.transform([feature_vector])
            
            # Make prediction
            prediction = self.model.predict(feature_vector_scaled)[0]
            predictions.append(max(0, prediction))
        
        return predictions
    
    def _calculate_seasonality(self, date: datetime) -> float:
        """Calculate seasonality factor"""
        month = date.month
        # Simple seasonality based on month (can be enhanced)
        seasonal_factors = {
            1: 1.1, 2: 1.0, 3: 0.9, 4: 0.8, 5: 0.9, 6: 1.0,
            7: 1.1, 8: 1.2, 9: 1.1, 10: 1.0, 11: 1.1, 12: 1.2
        }
        return seasonal_factors.get(month, 1.0)
    
    def _calculate_price_tier(self, price: float) -> int:
        """Calculate price tier (1-5)"""
        if price is None or price <= 0:
            return 1
        elif price <= 10:
            return 1
        elif price <= 25:
            return 2
        elif price <= 50:
            return 3
        elif price <= 100:
            return 4
        else:
            return 5
    
    def _encode_category(self, category: str) -> int:
        """Encode category to integer"""
        if not category:
            return 0
        
        if category not in self.label_encoders:
            self.label_encoders[category] = len(self.label_encoders) + 1
        
        return self.label_encoders[category]
    
    def _get_recent_consumption(self, medicine_id: int, days: int) -> pd.DataFrame:
        """Get recent consumption data for a medicine"""
        start_date = date.today() - timedelta(days=days)
        
        query = db.session.query(
            ConsumptionRecord.consumption_date,
            ConsumptionRecord.quantity_consumed
        ).filter(
            ConsumptionRecord.medicine_id == medicine_id,
            ConsumptionRecord.user_id == self.user_id,
            ConsumptionRecord.consumption_date >= start_date
        )
        
        return pd.read_sql(query.statement, db.engine)
    
    def _create_feature_vector(self, medicine: Medicine, target_date: date, consumption_data: pd.DataFrame) -> Optional[List[float]]:
        """Create feature vector for prediction"""
        try:
            # Calculate days until expiry
            days_until_expiry = (medicine.expiry_date - target_date).days
            
            # Calculate consumption rate
            if not consumption_data.empty:
                consumption_data['consumption_date'] = pd.to_datetime(consumption_data['consumption_date'])
                daily_consumption = consumption_data.groupby('consumption_date')['quantity_consumed'].sum().resample('D').sum().fillna(0)
                consumption_rate = daily_consumption.mean()
                recent_consumption = daily_consumption.tail(7).mean() if len(daily_consumption) >= 7 else consumption_rate
                weekly_consumption = daily_consumption.tail(14).head(7).mean() if len(daily_consumption) >= 14 else consumption_rate
            else:
                consumption_rate = 0.0
                recent_consumption = 0.0
                weekly_consumption = 0.0
            
            # Time-based features
            day_of_week = target_date.weekday()
            month = target_date.month
            seasonality = self._calculate_seasonality(datetime.combine(target_date, datetime.min.time()))
            
            # Medicine-specific features
            price_tier = self._calculate_price_tier(medicine.selling_price)
            category_encoded = self._encode_category(medicine.category)
            
            return [
                medicine.quantity,
                days_until_expiry,
                consumption_rate,
                seasonality,
                day_of_week,
                month,
                category_encoded,
                price_tier,
                recent_consumption,
                weekly_consumption
            ]
        except Exception as e:
            print(f"Error creating feature vector: {e}")
            return None
    
    def save_model(self) -> None:
        """Save trained model to disk"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'label_encoders': self.label_encoders,
            'feature_columns': self.feature_columns,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, f'models/demand_model_{self.user_id}.pkl')
    
    def load_model(self) -> bool:
        """Load trained model from disk"""
        try:
            model_data = joblib.load(f'models/demand_model_{self.user_id}.pkl')
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.label_encoders = model_data['label_encoders']
            self.feature_columns = model_data['feature_columns']
            self.is_trained = model_data['is_trained']
            return True
        except FileNotFoundError:
            return False


class AnomalyDetectionModel:
    """Machine Learning model for detecting anomalies in consumption patterns"""
    
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.model = IsolationForest(contamination=0.1, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def prepare_training_data(self, days: int = 90) -> pd.DataFrame:
        """Prepare training data for anomaly detection"""
        start_date = date.today() - timedelta(days=days)
        
        query = db.session.query(
            ConsumptionRecord.consumption_date,
            ConsumptionRecord.quantity_consumed,
            Medicine.id.label('medicine_id'),
            Medicine.category,
            Medicine.selling_price
        ).join(Medicine).filter(
            ConsumptionRecord.user_id == self.user_id,
            ConsumptionRecord.consumption_date >= start_date
        )
        
        df = pd.read_sql(query.statement, db.engine)
        
        if df.empty:
            return pd.DataFrame()
        
        # Create features for anomaly detection
        features = []
        
        for medicine_id in df['medicine_id'].unique():
            med_data = df[df['medicine_id'] == medicine_id]
            
            # Calculate consumption statistics
            consumption_stats = med_data['quantity_consumed'].describe()
            
            # Calculate time-based features
            med_data['consumption_date'] = pd.to_datetime(med_data['consumption_date'])
            daily_consumption = med_data.groupby('consumption_date')['quantity_consumed'].sum()
            
            if len(daily_consumption) < 7:
                continue
            
            # Calculate features
            mean_consumption = daily_consumption.mean()
            std_consumption = daily_consumption.std()
            max_consumption = daily_consumption.max()
            min_consumption = daily_consumption.min()
            consumption_frequency = len(daily_consumption) / days
            
            # Price tier
            price = med_data.iloc[0]['selling_price'] or 0
            price_tier = 1 if price <= 10 else 2 if price <= 25 else 3 if price <= 50 else 4 if price <= 100 else 5
            
            feature_vector = [
                mean_consumption,
                std_consumption,
                max_consumption,
                min_consumption,
                consumption_frequency,
                price_tier
            ]
            
            features.append(feature_vector)
        
        if not features:
            return pd.DataFrame()
        
        feature_df = pd.DataFrame(features, columns=[
            'mean_consumption', 'std_consumption', 'max_consumption',
            'min_consumption', 'consumption_frequency', 'price_tier'
        ])
        
        return feature_df
    
    def train(self) -> Dict[str, float]:
        """Train the anomaly detection model"""
        X = self.prepare_training_data()
        
        if X.empty or len(X) < 10:
            return {'error': 'Insufficient training data'}
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X)
        
        # Train model
        self.model.fit(X_scaled)
        
        # Detect anomalies in training data
        anomaly_scores = self.model.decision_function(X_scaled)
        predictions = self.model.predict(X_scaled)
        
        self.is_trained = True
        
        # Save model
        self.save_model()
        
        return {
            'training_samples': len(X),
            'anomalies_detected': sum(predictions == -1),
            'anomaly_rate': sum(predictions == -1) / len(X)
        }
    
    def detect_anomalies(self, days: int = 7) -> List[Dict]:
        """Detect anomalies in recent consumption patterns"""
        if not self.is_trained:
            self.load_model()
        
        if not self.is_trained:
            return []
        
        # Get recent consumption data
        start_date = date.today() - timedelta(days=days)
        
        query = db.session.query(
            ConsumptionRecord.consumption_date,
            ConsumptionRecord.quantity_consumed,
            Medicine.id.label('medicine_id'),
            Medicine.name,
            Medicine.category,
            Medicine.selling_price
        ).join(Medicine).filter(
            ConsumptionRecord.user_id == self.user_id,
            ConsumptionRecord.consumption_date >= start_date
        )
        
        df = pd.read_sql(query.statement, db.engine)
        
        if df.empty:
            return []
        
        anomalies = []
        
        for medicine_id in df['medicine_id'].unique():
            med_data = df[df['medicine_id'] == medicine_id]
            med_info = med_data.iloc[0]
            
            # Calculate features
            consumption_stats = med_data['quantity_consumed'].describe()
            mean_consumption = consumption_stats['mean']
            std_consumption = consumption_stats['std']
            max_consumption = consumption_stats['max']
            min_consumption = consumption_stats['min']
            consumption_frequency = len(med_data) / days
            
            price = med_info['selling_price'] or 0
            price_tier = 1 if price <= 10 else 2 if price <= 25 else 3 if price <= 50 else 4 if price <= 100 else 5
            
            feature_vector = [
                mean_consumption,
                std_consumption,
                max_consumption,
                min_consumption,
                consumption_frequency,
                price_tier
            ]
            
            # Scale features
            feature_vector_scaled = self.scaler.transform([feature_vector])
            
            # Predict anomaly
            prediction = self.model.predict(feature_vector_scaled)[0]
            anomaly_score = self.model.decision_function(feature_vector_scaled)[0]
            
            if prediction == -1:  # Anomaly detected
                anomalies.append({
                    'medicine_id': medicine_id,
                    'medicine_name': med_info['name'],
                    'category': med_info['category'],
                    'anomaly_score': float(anomaly_score),
                    'consumption_stats': {
                        'mean': float(mean_consumption),
                        'std': float(std_consumption),
                        'max': float(max_consumption),
                        'min': float(min_consumption),
                        'frequency': float(consumption_frequency)
                    }
                })
        
        return anomalies
    
    def save_model(self) -> None:
        """Save trained model to disk"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, f'models/anomaly_model_{self.user_id}.pkl')
    
    def load_model(self) -> bool:
        """Load trained model from disk"""
        try:
            model_data = joblib.load(f'models/anomaly_model_{self.user_id}.pkl')
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.is_trained = model_data['is_trained']
            return True
        except FileNotFoundError:
            return False


class ClusteringModel:
    """Machine Learning model for clustering medicines based on consumption patterns"""
    
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.model = KMeans(n_clusters=4, random_state=42)
        self.scaler = StandardScaler()
        self.is_trained = False
    
    def prepare_training_data(self, days: int = 90) -> pd.DataFrame:
        """Prepare training data for clustering"""
        start_date = date.today() - timedelta(days=days)
        
        query = db.session.query(
            ConsumptionRecord.consumption_date,
            ConsumptionRecord.quantity_consumed,
            Medicine.id.label('medicine_id'),
            Medicine.name,
            Medicine.category,
            Medicine.quantity,
            Medicine.selling_price
        ).join(Medicine).filter(
            ConsumptionRecord.user_id == self.user_id,
            ConsumptionRecord.consumption_date >= start_date
        )
        
        df = pd.read_sql(query.statement, db.engine)
        
        if df.empty:
            return pd.DataFrame()
        
        # Create features for clustering
        features = []
        medicine_info = []
        
        for medicine_id in df['medicine_id'].unique():
            med_data = df[df['medicine_id'] == medicine_id]
            med_info = med_data.iloc[0]
            
            # Calculate consumption features
            consumption_stats = med_data['quantity_consumed'].describe()
            
            # Calculate time-based features
            med_data['consumption_date'] = pd.to_datetime(med_data['consumption_date'])
            daily_consumption = med_data.groupby('consumption_date')['quantity_consumed'].sum()
            
            if len(daily_consumption) < 7:
                continue
            
            # Calculate features
            mean_consumption = daily_consumption.mean()
            std_consumption = daily_consumption.std()
            consumption_frequency = len(daily_consumption) / days
            consumption_consistency = 1 - (std_consumption / mean_consumption) if mean_consumption > 0 else 0
            
            # Price and quantity features
            price = med_info['selling_price'] or 0
            quantity = med_info['quantity']
            price_quantity_ratio = price / quantity if quantity > 0 else 0
            
            feature_vector = [
                mean_consumption,
                std_consumption,
                consumption_frequency,
                consumption_consistency,
                price,
                quantity,
                price_quantity_ratio
            ]
            
            features.append(feature_vector)
            medicine_info.append({
                'medicine_id': medicine_id,
                'name': med_info['name'],
                'category': med_info['category']
            })
        
        if not features:
            return pd.DataFrame()
        
        feature_df = pd.DataFrame(features, columns=[
            'mean_consumption', 'std_consumption', 'consumption_frequency',
            'consumption_consistency', 'price', 'quantity', 'price_quantity_ratio'
        ])
        
        # Add medicine info
        feature_df['medicine_info'] = medicine_info
        
        return feature_df
    
    def train(self) -> Dict[str, float]:
        """Train the clustering model"""
        X = self.prepare_training_data()
        
        if X.empty or len(X) < 4:
            return {'error': 'Insufficient training data'}
        
        # Extract features (exclude medicine_info)
        feature_columns = [col for col in X.columns if col != 'medicine_info']
        X_features = X[feature_columns]
        
        # Scale features
        X_scaled = self.scaler.fit_transform(X_features)
        
        # Train model
        self.model.fit(X_scaled)
        
        # Get cluster assignments
        cluster_labels = self.model.labels_
        
        self.is_trained = True
        
        # Save model
        self.save_model()
        
        # Analyze clusters
        cluster_analysis = self._analyze_clusters(X, cluster_labels)
        
        return {
            'training_samples': len(X),
            'clusters': len(set(cluster_labels)),
            'cluster_analysis': cluster_analysis
        }
    
    def _analyze_clusters(self, X: pd.DataFrame, cluster_labels: np.ndarray) -> Dict:
        """Analyze cluster characteristics"""
        analysis = {}
        
        for cluster_id in set(cluster_labels):
            cluster_data = X[cluster_labels == cluster_id]
            
            analysis[cluster_id] = {
                'size': len(cluster_data),
                'mean_consumption': cluster_data['mean_consumption'].mean(),
                'consumption_frequency': cluster_data['consumption_frequency'].mean(),
                'price_range': {
                    'min': cluster_data['price'].min(),
                    'max': cluster_data['price'].max(),
                    'mean': cluster_data['price'].mean()
                },
                'quantity_range': {
                    'min': cluster_data['quantity'].min(),
                    'max': cluster_data['quantity'].max(),
                    'mean': cluster_data['quantity'].mean()
                }
            }
        
        return analysis
    
    def predict_clusters(self) -> List[Dict]:
        """Predict clusters for all medicines"""
        if not self.is_trained:
            self.load_model()
        
        if not self.is_trained:
            return []
        
        X = self.prepare_training_data()
        
        if X.empty:
            return []
        
        # Extract features
        feature_columns = [col for col in X.columns if col != 'medicine_info']
        X_features = X[feature_columns]
        
        # Scale features
        X_scaled = self.scaler.transform(X_features)
        
        # Predict clusters
        cluster_labels = self.model.predict(X_scaled)
        
        # Create results
        results = []
        for i, (_, row) in enumerate(X.iterrows()):
            results.append({
                'medicine_id': row['medicine_info']['medicine_id'],
                'name': row['medicine_info']['name'],
                'category': row['medicine_info']['category'],
                'cluster': int(cluster_labels[i]),
                'features': {
                    'mean_consumption': float(row['mean_consumption']),
                    'consumption_frequency': float(row['consumption_frequency']),
                    'price': float(row['price']),
                    'quantity': int(row['quantity'])
                }
            })
        
        return results
    
    def save_model(self) -> None:
        """Save trained model to disk"""
        model_data = {
            'model': self.model,
            'scaler': self.scaler,
            'is_trained': self.is_trained
        }
        
        joblib.dump(model_data, f'models/clustering_model_{self.user_id}.pkl')
    
    def load_model(self) -> bool:
        """Load trained model from disk"""
        try:
            model_data = joblib.load(f'models/clustering_model_{self.user_id}.pkl')
            self.model = model_data['model']
            self.scaler = model_data['scaler']
            self.is_trained = model_data['is_trained']
            return True
        except FileNotFoundError:
            return False


class MLModelManager:
    """Manager class for all ML models"""
    
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.demand_model = DemandForecastingModel(user_id)
        self.anomaly_model = AnomalyDetectionModel(user_id)
        self.clustering_model = ClusteringModel(user_id)
    
    def train_all_models(self) -> Dict[str, Dict]:
        """Train all ML models"""
        results = {}
        
        # Train demand forecasting model
        try:
            results['demand_forecasting'] = self.demand_model.train()
        except Exception as e:
            results['demand_forecasting'] = {'error': str(e)}
        
        # Train anomaly detection model
        try:
            results['anomaly_detection'] = self.anomaly_model.train()
        except Exception as e:
            results['anomaly_detection'] = {'error': str(e)}
        
        # Train clustering model
        try:
            results['clustering'] = self.clustering_model.train()
        except Exception as e:
            results['clustering'] = {'error': str(e)}
        
        return results
    
    def get_ml_insights(self) -> Dict:
        """Get insights from all ML models"""
        insights = {}
        
        # Demand forecasting insights
        try:
            if self.demand_model.is_trained:
                insights['demand_forecasting'] = {
                    'status': 'trained',
                    'model_type': 'Random Forest Regressor'
                }
            else:
                insights['demand_forecasting'] = {'status': 'not_trained'}
        except Exception as e:
            insights['demand_forecasting'] = {'status': 'error', 'error': str(e)}
        
        # Anomaly detection insights
        try:
            if self.anomaly_model.is_trained:
                anomalies = self.anomaly_model.detect_anomalies()
                insights['anomaly_detection'] = {
                    'status': 'trained',
                    'anomalies_detected': len(anomalies),
                    'recent_anomalies': anomalies[:5]  # Top 5 anomalies
                }
            else:
                insights['anomaly_detection'] = {'status': 'not_trained'}
        except Exception as e:
            insights['anomaly_detection'] = {'status': 'error', 'error': str(e)}
        
        # Clustering insights
        try:
            if self.clustering_model.is_trained:
                clusters = self.clustering_model.predict_clusters()
                insights['clustering'] = {
                    'status': 'trained',
                    'total_medicines': len(clusters),
                    'clusters': len(set(c['cluster'] for c in clusters))
                }
            else:
                insights['clustering'] = {'status': 'not_trained'}
        except Exception as e:
            insights['clustering'] = {'status': 'error', 'error': str(e)}
        
        return insights
