from datetime import date, timedelta, datetime
from typing import List, Optional, Dict, Tuple
from flask import current_app
from . import db
from .models import Medicine, User, ConsumptionRecord, InventoryAlert, AnalyticsCache
from .ds.structures import MinExpiryHeap, NameHashMap, MedicineNode


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
        # Remove from in-memory index
        try:
            inventory_index.remove_medicine(med)
        except Exception:
            pass

        # Delete dependent consumption records and alerts explicitly to avoid
        # SQLAlchemy attempting to set FK to NULL on delete (which fails when
        # the FK is NOT NULL in SQLite). Use bulk delete for efficiency.
        try:
            db.session.query(ConsumptionRecord).filter(ConsumptionRecord.medicine_id == med.id).delete(synchronize_session=False)
        except Exception:
            pass
        try:
            db.session.query(InventoryAlert).filter(InventoryAlert.medicine_id == med.id).delete(synchronize_session=False)
        except Exception:
            pass

        db.session.delete(med)
        count += 1
    if count:
        db.session.commit()
    return count


# Simple inventory service (analytics removed)
class SimpleInventoryService:
    """Simple inventory management service"""
    
    def __init__(self, user_id: int):
        self.user_id = user_id
    
    def get_dashboard_analytics(self) -> Dict:
        """Get simple dashboard analytics"""
        medicines = Medicine.query.filter(Medicine.user_id == self.user_id).all()
        
        if not medicines:
            return {
                'insights': {
                    'total_inventory_value': 0,
                    'low_stock_count': 0,
                    'risk_distribution': {'high_risk': 0, 'medium_risk': 0, 'low_risk': 0},
                    'categories': {}
                },
                'recommendations': [],
                'consumption_trends': {'trend': 'stable', 'change_percentage': 0},
                'stock_analysis': {'low_stock_count': 0, 'overstocked_count': 0},
                'expiry_analysis': {'expiring_7_days_count': 0, 'total_at_risk_value': 0.0}
            }
        
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
            'insights': {
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
                }
            },
            'recommendations': [],
            'consumption_trends': {'trend': 'stable', 'change_percentage': 0},
            'stock_analysis': {
                'low_stock_count': len(low_stock),
                'overstocked_count': len(overstocked),
                'optimal_stock_count': total_medicines - len(low_stock) - len(overstocked)
            },
            'expiry_analysis': {
                'expiring_7_days_count': len([m for m in medicines if 0 < m.days_until_expiry() <= 7]),
                'total_at_risk_value': sum(m.quantity * (m.selling_price or 0) for m in expiring_soon)
            }
        }
    
    def create_consumption_record(self, medicine_id: int, quantity: int, notes: str = None) -> ConsumptionRecord:
        """Create consumption record"""
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
        
        db.session.commit()
        return record

    def generate_alerts(self) -> None:
        """Generate inventory alerts for the user and persist them.

        This creates alerts for expired items, items expiring soon (7 days),
        low stock, and overstocked items. It avoids creating duplicate active
        alerts by checking for existing unread alerts of the same type.
        """
        meds = Medicine.query.filter(Medicine.user_id == self.user_id).all()
        for med in meds:
            now_alerts = []
            # Determine alert types applicable
            if med.is_expired():
                now_alerts.append(('expired', f'{med.name} has expired'))
            elif 0 < med.days_until_expiry() <= 7:
                now_alerts.append(('expiring_soon', f'{med.name} is expiring soon'))

            if med.is_low_stock():
                now_alerts.append(('low_stock', f'{med.name} is low on stock'))

            if med.is_overstocked():
                now_alerts.append(('overstocked', f'{med.name} appears overstocked'))

            for alert_type, message in now_alerts:
                exists = (
                    InventoryAlert.query.filter(
                        InventoryAlert.user_id == self.user_id,
                        InventoryAlert.medicine_id == med.id,
                        InventoryAlert.alert_type == alert_type,
                        InventoryAlert.is_read == False,
                    ).first()
                    is not None
                )
                if not exists:
                    a = InventoryAlert(
                        user_id=self.user_id,
                        medicine_id=med.id,
                        alert_type=alert_type,
                        message=message,
                        severity='high' if alert_type in ('expired', 'low_stock') else 'medium',
                    )
                    db.session.add(a)
        try:
            db.session.commit()
        except Exception:
            db.session.rollback()


# Global service instances
simple_services = {}  # user_id -> SimpleInventoryService

def get_advanced_service(user_id: int) -> SimpleInventoryService:
    """Get or create simple service for user"""
    if user_id not in simple_services:
        simple_services[user_id] = SimpleInventoryService(user_id)
    return simple_services[user_id]


