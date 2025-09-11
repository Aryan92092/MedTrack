from datetime import date, timedelta, datetime
from typing import List, Optional, Dict, Tuple
from flask import current_app
from . import db
from .models import Medicine, User, ConsumptionRecord, InventoryAlert, AnalyticsCache
from .ds.structures import MinExpiryHeap, NameHashMap, MedicineNode
from .ds.analytics import InventoryAnalytics, PredictiveAnalytics
import pandas as pd
import numpy as np


class InventoryIndex:
    def __init__(self) -> None:
        self.heap = MinExpiryHeap()
        self.name_index = NameHashMap()

    def rebuild(self, user_id: int | None = None) -> None:
        self.heap = MinExpiryHeap()
        self.name_index = NameHashMap()
        query = Medicine.query
        if user_id is not None:
            # Only current user's items
            query = query.filter(Medicine.user_id == user_id)
        for med in query.all():
            node = MedicineNode(
                id=med.id,
                name=med.name,
                quantity=med.quantity,
                expiry_date=med.expiry_date,
            )
            self.heap.push(node)
            self.name_index.add(med.name, med.id)

    def add_medicine(self, med: Medicine) -> None:
        node = MedicineNode(
            id=med.id,
            name=med.name,
            quantity=med.quantity,
            expiry_date=med.expiry_date,
        )
        self.heap.push(node)
        self.name_index.add(med.name, med.id)

    def update_medicine(self, med: Medicine) -> None:
        node = MedicineNode(
            id=med.id,
            name=med.name,
            quantity=med.quantity,
            expiry_date=med.expiry_date,
        )
        self.heap.update(node)
        # ensure name index reflects potential rename
        # simple approach: rebuild name index entry
        for key, ids in list(self.name_index._name_to_ids.items()):
            if med.id in ids and key != med.name.lower():
                self.name_index.remove(key, med.id)
        self.name_index.add(med.name, med.id)

    def remove_medicine(self, med: Medicine) -> None:
        self.heap.remove(med.id)
        self.name_index.remove(med.name, med.id)

    def soon_expiring(self, within_days: int = 30) -> List[Medicine]:
        today = date.today()
        threshold = today + timedelta(days=within_days)
        ids: List[int] = []
        for node in self.heap.as_sorted_list():
            if node.expiry_date <= threshold:
                ids.append(node.id)
        if not ids:
            return []
        return (
            Medicine.query.filter(Medicine.id.in_(ids))
            .order_by(Medicine.expiry_date.asc())
            .all()
        )

    def sorted_by_expiry(self) -> List[Medicine]:
        ids = [n.id for n in self.heap.as_sorted_list()]
        if not ids:
            return []
        # preserve order using CASE
        cases = {mid: idx for idx, mid in enumerate(ids)}
        meds = Medicine.query.filter(Medicine.id.in_(ids)).all()
        meds.sort(key=lambda m: cases.get(m.id, 10**9))
        return meds

    def search_by_name(self, name: str) -> List[Medicine]:
        ids = self.name_index.get_ids(name)
        if not ids:
            return []
        meds = Medicine.query.filter(Medicine.id.in_(ids)).all()
        # prefer earliest expiry first
        meds.sort(key=lambda m: m.expiry_date)
        return meds


inventory_index = InventoryIndex()


def cleanup_expired(user_id: int | None = None) -> int:
    today = date.today()
    query = Medicine.query.filter(Medicine.expiry_date < today)
    if user_id is not None:
        query = query.filter(Medicine.user_id == user_id)
    expired = query.all()
    count = 0
    for med in expired:
        inventory_index.remove_medicine(med)
        db.session.delete(med)
        count += 1
    if count:
        db.session.commit()
    return count


# Advanced Data Science Services
class AdvancedInventoryService:
    """Advanced inventory management with data science capabilities"""
    
    def __init__(self, user_id: int):
        self.user_id = user_id
        self.analytics = InventoryAnalytics(user_id)
        self.predictive = PredictiveAnalytics(user_id)
    
    def get_dashboard_analytics(self) -> Dict:
        """Get comprehensive dashboard analytics"""
        cache_key = f"dashboard_analytics_{self.user_id}"
        cached_data = self.analytics.get_cached_analytics(cache_key)
        
        if cached_data:
            return cached_data
        
        # Generate fresh analytics
        insights = self.analytics.generate_inventory_insights()
        recommendations = self.analytics.generate_recommendations()
        
        # Get recent consumption trends
        consumption_data = self.analytics.get_consumption_data(30)
        consumption_trends = self._calculate_consumption_trends(consumption_data)
        
        # Get stock level analysis
        stock_analysis = self._analyze_stock_levels()
        
        # Get expiry analysis
        expiry_analysis = self._analyze_expiry_patterns()
        
        analytics_data = {
            'insights': insights,
            'recommendations': recommendations,
            'consumption_trends': consumption_trends,
            'stock_analysis': stock_analysis,
            'expiry_analysis': expiry_analysis,
            'generated_at': datetime.utcnow().isoformat()
        }
        
        # Cache the results
        self.analytics.cache_analytics(cache_key, analytics_data, hours=6)
        
        return analytics_data
    
    def _calculate_consumption_trends(self, consumption_data: pd.DataFrame) -> Dict:
        """Calculate consumption trends"""
        if consumption_data.empty:
            return {'trend': 'stable', 'change_percentage': 0}
        
        # Group by week
        consumption_data['week'] = pd.to_datetime(consumption_data['consumption_date']).dt.to_period('W')
        weekly_consumption = consumption_data.groupby('week')['quantity_consumed'].sum()
        
        if len(weekly_consumption) < 2:
            return {'trend': 'stable', 'change_percentage': 0}
        
        # Calculate trend
        recent_weeks = weekly_consumption.tail(4)
        older_weeks = weekly_consumption.head(-4) if len(weekly_consumption) > 4 else weekly_consumption.head(2)
        
        recent_avg = recent_weeks.mean()
        older_avg = older_weeks.mean()
        
        change_percentage = ((recent_avg - older_avg) / older_avg * 100) if older_avg > 0 else 0
        
        if change_percentage > 10:
            trend = 'increasing'
        elif change_percentage < -10:
            trend = 'decreasing'
        else:
            trend = 'stable'
        
        return {
            'trend': trend,
            'change_percentage': round(change_percentage, 2),
            'recent_weekly_average': round(recent_avg, 2),
            'previous_weekly_average': round(older_avg, 2)
        }
    
    def _analyze_stock_levels(self) -> Dict:
        """Analyze stock levels and provide insights"""
        medicines = Medicine.query.filter(Medicine.user_id == self.user_id).all()
        
        if not medicines:
            return {}
        
        low_stock = [m for m in medicines if m.is_low_stock()]
        overstocked = [m for m in medicines if m.is_overstocked()]
        optimal_stock = [m for m in medicines if not m.is_low_stock() and not m.is_overstocked()]
        
        # Calculate stock value distribution
        total_value = sum(m.quantity * (m.selling_price or 0) for m in medicines)
        low_stock_value = sum(m.quantity * (m.selling_price or 0) for m in low_stock)
        overstocked_value = sum(m.quantity * (m.selling_price or 0) for m in overstocked)
        
        return {
            'low_stock_count': len(low_stock),
            'overstocked_count': len(overstocked),
            'optimal_stock_count': len(optimal_stock),
            'total_value': total_value,
            'low_stock_value': low_stock_value,
            'overstocked_value': overstocked_value,
            'low_stock_percentage': (low_stock_value / total_value * 100) if total_value > 0 else 0,
            'overstocked_percentage': (overstocked_value / total_value * 100) if total_value > 0 else 0
        }
    
    def _analyze_expiry_patterns(self) -> Dict:
        """Analyze expiry patterns and risks"""
        medicines = Medicine.query.filter(Medicine.user_id == self.user_id).all()
        
        if not medicines:
            return {}
        
        today = date.today()
        expired = [m for m in medicines if m.is_expired()]
        expiring_7_days = [m for m in medicines if 0 < m.days_until_expiry() <= 7]
        expiring_30_days = [m for m in medicines if 7 < m.days_until_expiry() <= 30]
        expiring_90_days = [m for m in medicines if 30 < m.days_until_expiry() <= 90]
        
        # Calculate total value at risk
        expired_value = sum(m.quantity * (m.selling_price or 0) for m in expired)
        expiring_7_value = sum(m.quantity * (m.selling_price or 0) for m in expiring_7_days)
        expiring_30_value = sum(m.quantity * (m.selling_price or 0) for m in expiring_30_days)
        
        return {
            'expired_count': len(expired),
            'expiring_7_days_count': len(expiring_7_days),
            'expiring_30_days_count': len(expiring_30_days),
            'expiring_90_days_count': len(expiring_90_days),
            'expired_value': expired_value,
            'expiring_7_value': expiring_7_value,
            'expiring_30_value': expiring_30_value,
            'total_at_risk_value': expired_value + expiring_7_value + expiring_30_value
        }
    
    def generate_smart_recommendations(self) -> List[Dict]:
        """Generate smart recommendations using ML and analytics"""
        recommendations = []
        medicines = Medicine.query.filter(Medicine.user_id == self.user_id).all()
        
        # Get consumption rates
        consumption_rates = self.analytics.calculate_consumption_rates()
        
        for med in medicines:
            consumption_rate = consumption_rates.get(med.id, 0)
            
            # Smart reorder recommendations
            if consumption_rate > 0 and not med.is_expired():
                days_until_stockout = med.quantity / consumption_rate if consumption_rate > 0 else float('inf')
                
                if days_until_stockout <= 7:
                    recommendations.append({
                        'type': 'urgent_reorder',
                        'medicine_id': med.id,
                        'medicine_name': med.name,
                        'message': f'{med.name} will run out in {days_until_stockout:.1f} days',
                        'priority': 'high',
                        'action': 'reorder_immediately',
                        'suggested_quantity': max(med.min_stock_level, int(consumption_rate * 30))
                    })
                elif days_until_stockout <= 14:
                    recommendations.append({
                        'type': 'reorder',
                        'medicine_id': med.id,
                        'medicine_name': med.name,
                        'message': f'{med.name} will run out in {days_until_stockout:.1f} days',
                        'priority': 'medium',
                        'action': 'reorder_soon',
                        'suggested_quantity': max(med.min_stock_level, int(consumption_rate * 30))
                    })
            
            # Expiry optimization
            if med.days_until_expiry() <= 30 and not med.is_expired():
                if consumption_rate > 0:
                    days_to_consume = med.quantity / consumption_rate
                    if days_to_consume > med.days_until_expiry():
                        recommendations.append({
                            'type': 'expiry_risk',
                            'medicine_id': med.id,
                            'medicine_name': med.name,
                            'message': f'{med.name} may expire before being fully consumed',
                            'priority': 'high',
                            'action': 'promote_usage',
                            'days_until_expiry': med.days_until_expiry()
                        })
            
            # Overstock optimization
            if med.is_overstocked() and consumption_rate > 0:
                months_of_stock = med.quantity / (consumption_rate * 30)
                if months_of_stock > 6:
                    recommendations.append({
                        'type': 'overstock',
                        'medicine_id': med.id,
                        'medicine_name': med.name,
                        'message': f'{med.name} has {months_of_stock:.1f} months of stock',
                        'priority': 'low',
                        'action': 'reduce_stock',
                        'suggested_quantity': int(consumption_rate * 90)  # 3 months supply
                    })
        
        return sorted(recommendations, key=lambda x: ['urgent_reorder', 'expiry_risk', 'reorder', 'overstock'].index(x['type']))
    
    def create_consumption_record(self, medicine_id: int, quantity: int, notes: str = None) -> ConsumptionRecord:
        """Create consumption record and update analytics"""
        return self.analytics.create_consumption_record(medicine_id, quantity, notes)
    
    def get_consumption_analytics(self, days: int = 30) -> Dict:
        """Get detailed consumption analytics"""
        consumption_data = self.analytics.get_consumption_data(days)
        
        if consumption_data.empty:
            return {'total_consumption': 0, 'daily_average': 0, 'trends': {}}
        
        # Calculate metrics
        total_consumption = consumption_data['quantity_consumed'].sum()
        daily_average = total_consumption / days
        
        # Top consumed medicines
        top_medicines = consumption_data.groupby('medicine_id')['quantity_consumed'].sum().nlargest(10)
        
        # Daily trends
        daily_trends = consumption_data.groupby('consumption_date')['quantity_consumed'].sum()
        
        return {
            'total_consumption': int(total_consumption),
            'daily_average': round(daily_average, 2),
            'top_medicines': top_medicines.to_dict(),
            'daily_trends': daily_trends.to_dict(),
            'period_days': days
        }
    
    def generate_alerts(self) -> List[InventoryAlert]:
        """Generate system alerts based on analytics"""
        alerts = []
        medicines = Medicine.query.filter(Medicine.user_id == self.user_id).all()
        
        for med in medicines:
            # Expired medicines
            if med.is_expired():
                alert = InventoryAlert(
                    user_id=self.user_id,
                    medicine_id=med.id,
                    alert_type='expired',
                    message=f'{med.name} has expired and should be removed',
                    severity='critical'
                )
                alerts.append(alert)
            
            # Low stock alerts
            elif med.is_low_stock():
                alert = InventoryAlert(
                    user_id=self.user_id,
                    medicine_id=med.id,
                    alert_type='low_stock',
                    message=f'{med.name} is running low (quantity: {med.quantity})',
                    severity='high'
                )
                alerts.append(alert)
            
            # Expiring soon alerts
            elif med.days_until_expiry() <= 7:
                alert = InventoryAlert(
                    user_id=self.user_id,
                    medicine_id=med.id,
                    alert_type='expiring_soon',
                    message=f'{med.name} expires in {med.days_until_expiry()} days',
                    severity='high'
                )
                alerts.append(alert)
            
            # High risk alerts
            elif med.calculate_risk_score() > 0.8:
                alert = InventoryAlert(
                    user_id=self.user_id,
                    medicine_id=med.id,
                    alert_type='high_risk',
                    message=f'{med.name} has high risk score ({med.calculate_risk_score():.2f})',
                    severity='medium'
                )
                alerts.append(alert)
        
        # Save alerts to database
        for alert in alerts:
            db.session.add(alert)
        db.session.commit()
        
        return alerts


# Global service instances
advanced_services = {}  # user_id -> AdvancedInventoryService

def get_advanced_service(user_id: int) -> AdvancedInventoryService:
    """Get or create advanced service for user"""
    if user_id not in advanced_services:
        advanced_services[user_id] = AdvancedInventoryService(user_id)
    return advanced_services[user_id]


