from datetime import date, datetime
from flask_login import UserMixin
from werkzeug.security import generate_password_hash, check_password_hash
from . import db, login_manager
import json


class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    username = db.Column(db.String(80), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=False)
    is_admin = db.Column(db.Boolean, default=True)
    # Profile fields
    full_name = db.Column(db.String(120), nullable=True)
    email = db.Column(db.String(120), unique=False, nullable=True, index=True)
    phone = db.Column(db.String(30), nullable=True)
    profile_pic = db.Column(db.String(255), nullable=True)

    def set_password(self, password: str) -> None:
        self.password_hash = generate_password_hash(password)

    def check_password(self, password: str) -> bool:
        return check_password_hash(self.password_hash, password)


@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))


class Medicine(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(120), index=True, nullable=False)
    quantity = db.Column(db.Integer, nullable=False, default=0)
    expiry_date = db.Column(db.Date, nullable=False)
    # Owner of this record. Nullable for backward compatibility; new records set this.
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    
    # Enhanced DS fields
    category = db.Column(db.String(50), nullable=True, index=True)
    manufacturer = db.Column(db.String(100), nullable=True)
    batch_number = db.Column(db.String(50), nullable=True)
    purchase_price = db.Column(db.Float, nullable=True)
    selling_price = db.Column(db.Float, nullable=True)
    min_stock_level = db.Column(db.Integer, default=10)
    max_stock_level = db.Column(db.Integer, default=1000)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow, onupdate=datetime.utcnow)
    
    # Analytics fields
    consumption_rate = db.Column(db.Float, default=0.0)  # units per day
    demand_forecast = db.Column(db.Float, default=0.0)  # predicted demand
    risk_score = db.Column(db.Float, default=0.0)  # 0-1 risk of stockout/expiry
    last_consumption_date = db.Column(db.Date, nullable=True)

    def is_expired(self) -> bool:
        return self.expiry_date < date.today()
    
    def days_until_expiry(self) -> int:
        return (self.expiry_date - date.today()).days
    
    def is_low_stock(self) -> bool:
        return self.quantity <= self.min_stock_level
    
    def is_overstocked(self) -> bool:
        return self.quantity >= self.max_stock_level
    
    def calculate_risk_score(self) -> float:
        """Calculate risk score based on expiry, stock levels, and consumption rate"""
        # Compute days until expiry (may be negative if already expired)
        days_until_expiry = self.days_until_expiry()

        # Coerce nullable numeric fields to safe defaults
        max_stock = self.max_stock_level if (self.max_stock_level is not None) else 1
        quantity = self.quantity if (self.quantity is not None) else 0
        consumption_rate = self.consumption_rate if (self.consumption_rate is not None) else 0.0

        # Prevent division by zero
        try:
            stock_ratio = quantity / max(max_stock, 1)
        except Exception:
            stock_ratio = 0.0

        # Expiry risk (higher as expiry approaches)
        expiry_risk = max(0.0, 1 - (days_until_expiry / 30)) if days_until_expiry > 0 else 1.0

        # Stockout risk (higher as stock decreases)
        stockout_risk = max(0.0, 1 - stock_ratio)

        # Consumption risk (higher if consumption rate is high and stock is low)
        consumption_risk = min(1.0, (consumption_rate * 7) / max(quantity, 1)) if quantity > 0 else 0.0

        # Weighted combination
        risk_score = (expiry_risk * 0.4 + stockout_risk * 0.4 + consumption_risk * 0.2)

        # Persist and return a bounded score
        self.risk_score = min(1.0, max(0.0, risk_score))
        return self.risk_score
    
    def to_dict(self) -> dict:
        """Convert medicine to dictionary for JSON serialization"""
        return {
            'id': self.id,
            'name': self.name,
            'quantity': self.quantity,
            'expiry_date': self.expiry_date.isoformat(),
            'category': self.category,
            'manufacturer': self.manufacturer,
            'batch_number': self.batch_number,
            'purchase_price': self.purchase_price,
            'selling_price': self.selling_price,
            'min_stock_level': self.min_stock_level,
            'max_stock_level': self.max_stock_level,
            'consumption_rate': self.consumption_rate,
            'demand_forecast': self.demand_forecast,
            'risk_score': self.risk_score,
            'is_expired': self.is_expired(),
            'days_until_expiry': self.days_until_expiry(),
            'is_low_stock': self.is_low_stock(),
            'is_overstocked': self.is_overstocked()
        }


# Analytics and Consumption Tracking Models
class ConsumptionRecord(db.Model):
    """Track medicine consumption for analytics"""
    id = db.Column(db.Integer, primary_key=True)
    medicine_id = db.Column(db.Integer, db.ForeignKey('medicine.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    quantity_consumed = db.Column(db.Integer, nullable=False)
    consumption_date = db.Column(db.Date, nullable=False, default=date.today)
    notes = db.Column(db.Text, nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    medicine = db.relationship('Medicine', backref='consumption_records')
    user = db.relationship('User', backref='consumption_records')


class AnalyticsCache(db.Model):
    """Cache for expensive analytics calculations"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    cache_key = db.Column(db.String(100), nullable=False, index=True)
    cache_data = db.Column(db.Text, nullable=False)  # JSON data
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    expires_at = db.Column(db.DateTime, nullable=False)
    
    def is_expired(self) -> bool:
        return datetime.utcnow() > self.expires_at
    
    def get_data(self) -> dict:
        return json.loads(self.cache_data)
    
    def set_data(self, data: dict) -> None:
        self.cache_data = json.dumps(data)


class InventoryAlert(db.Model):
    """System-generated alerts for inventory management"""
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    medicine_id = db.Column(db.Integer, db.ForeignKey('medicine.id'), nullable=True)
    alert_type = db.Column(db.String(50), nullable=False)  # 'low_stock', 'expiring_soon', 'overstocked', 'expired'
    message = db.Column(db.Text, nullable=False)
    severity = db.Column(db.String(20), default='medium')  # 'low', 'medium', 'high', 'critical'
    is_read = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Relationships
    user = db.relationship('User', backref='alerts')
    medicine = db.relationship('Medicine', backref='alerts')


# Optional helper relationship for convenience
User.medicines = db.relationship('Medicine', backref='owner', lazy=True)


