from datetime import datetime, date
from flask import Blueprint, render_template, redirect, url_for, request, flash, jsonify
from flask_login import login_required, current_user
from ..models import Medicine, User, ConsumptionRecord, InventoryAlert
from .. import db
from ..services import inventory_index, cleanup_expired, get_advanced_service
from ..ds.ml_models import MLModelManager
from sqlalchemy import or_
import json
from types import SimpleNamespace


def _to_plain_dict(obj):
    """Recursively convert SimpleNamespace/objects to plain dicts/lists so they are JSON-serializable."""
    if obj is None:
        return None
    if isinstance(obj, dict):
        return {k: _to_plain_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_plain_dict(v) for v in obj]
    # SimpleNamespace or objects with __dict__
    if isinstance(obj, SimpleNamespace):
        return _to_plain_dict(vars(obj))
    if hasattr(obj, '__dict__'):
        return _to_plain_dict(vars(obj))
    return obj


def _ensure_analytics_defaults(analytics_obj):
    """Return a plain dict with safe defaults for analytics used in templates.

    Accepts dicts or objects (SimpleNamespace) and always returns a JSON-serializable dict.
    """
    analytics = _to_plain_dict(analytics_obj) if not isinstance(analytics_obj, dict) else dict(analytics_obj)

    if analytics is None:
        analytics = {}

    insights = analytics.get('insights') or {}
    # ensure expected insight fields
    insights.setdefault('total_inventory_value', 0.0)
    insights.setdefault('low_stock_count', 0)
    insights.setdefault('risk_distribution', {'high_risk': 0, 'medium_risk': 0, 'low_risk': 0})
    insights.setdefault('categories', {})
    analytics['insights'] = insights

    consumption_trends = analytics.get('consumption_trends') or {}
    consumption_trends.setdefault('trend', 'stable')
    consumption_trends.setdefault('change_percentage', 0)
    analytics['consumption_trends'] = consumption_trends

    stock_analysis = analytics.get('stock_analysis') or {}
    stock_analysis.setdefault('low_stock_count', 0)
    stock_analysis.setdefault('overstocked_count', 0)
    analytics['stock_analysis'] = stock_analysis

    expiry_analysis = analytics.get('expiry_analysis') or {}
    expiry_analysis.setdefault('expiring_7_days_count', 0)
    expiry_analysis.setdefault('total_at_risk_value', 0.0)
    analytics['expiry_analysis'] = expiry_analysis

    analytics.setdefault('recommendations', [])
    analytics.setdefault('consumption_trends', consumption_trends)
    analytics.setdefault('stock_analysis', stock_analysis)
    analytics.setdefault('expiry_analysis', expiry_analysis)

    return analytics


main_bp = Blueprint('main', __name__)


@main_bp.before_request
def ensure_index():
    # Rebuild per user session when empty
    if len(inventory_index.heap) == 0:
        inventory_index.rebuild(user_id=current_user.get_id() if current_user.is_authenticated else None)


@main_bp.route('/')
@login_required
def dashboard():
    user_id = int(current_user.get_id())
    
    # Clean up expired medicines
    cleanup_count = cleanup_expired(user_id=user_id)
    
    # Get advanced analytics
    advanced_service = get_advanced_service(user_id)
    analytics = advanced_service.get_dashboard_analytics()
    # templates expect attribute access (analytics.insights.total_inventory_value)
    if not isinstance(analytics, dict):
        analytics = _to_plain_dict(analytics)
    analytics = _ensure_analytics_defaults(analytics)
    
    # Get basic data for backward compatibility
    soon = inventory_index.soon_expiring(within_days=30)
    total = Medicine.query.filter(Medicine.user_id == user_id).count()
    
    # Get recent alerts
    recent_alerts = InventoryAlert.query.filter(
        InventoryAlert.user_id == user_id,
        InventoryAlert.is_read == False
    ).order_by(InventoryAlert.created_at.desc()).limit(5).all()
    
    return render_template('dashboard.html', 
                         soon=soon, 
                         total=total, 
                         cleanup_count=cleanup_count,
                         analytics=analytics,
                         alerts=recent_alerts)


@main_bp.route('/profile', methods=['GET', 'POST'])
@login_required
def profile():
    user: User = User.query.get_or_404(int(current_user.get_id()))
    if request.method == 'POST':
        user.full_name = request.form.get('full_name', '').strip() or None
        user.email = request.form.get('email', '').strip() or None
        user.phone = request.form.get('phone', '').strip() or None
        # Handle profile picture URL input for simplicity (file uploads can be added)
        pic = request.form.get('profile_pic', '').strip()
        user.profile_pic = pic or user.profile_pic
        db.session.commit()
        flash('Profile updated', 'success')
        return redirect(url_for('main.profile'))
    return render_template('profile.html', user=user)


@main_bp.route('/medicines')
@login_required
def medicines():
    meds = inventory_index.sorted_by_expiry()
    return render_template('medicines.html', medicines=meds)


@main_bp.route('/add', methods=['GET', 'POST'])
@login_required
def add_medicine():
    if request.method == 'POST':
        name = request.form.get('name', '').strip()
        quantity = int(request.form.get('quantity', '0') or 0)
        expiry_str = request.form.get('expiry_date', '')
        
        # Enhanced fields
        category = request.form.get('category', '').strip() or None
        manufacturer = request.form.get('manufacturer', '').strip() or None
        batch_number = request.form.get('batch_number', '').strip() or None
        purchase_price = float(request.form.get('purchase_price', '0') or 0) or None
        selling_price = float(request.form.get('selling_price', '0') or 0) or None
        min_stock_level = int(request.form.get('min_stock_level', '10') or 10)
        max_stock_level = int(request.form.get('max_stock_level', '1000') or 1000)
        
        try:
            expiry = datetime.strptime(expiry_str, '%Y-%m-%d').date()
        except ValueError:
            flash('Invalid expiry date', 'danger')
            return render_template('add.html')
        
        if not name or quantity <= 0:
            flash('Please provide a name and positive quantity', 'danger')
            return render_template('add.html')
        
        if min_stock_level >= max_stock_level:
            flash('Minimum stock level must be less than maximum stock level', 'danger')
            return render_template('add.html')
        
        med = Medicine(
            name=name, 
            quantity=quantity, 
            expiry_date=expiry, 
            user_id=int(current_user.get_id()),
            category=category,
            manufacturer=manufacturer,
            batch_number=batch_number,
            purchase_price=purchase_price,
            selling_price=selling_price,
            min_stock_level=min_stock_level,
            max_stock_level=max_stock_level
        )
        
        # Calculate initial risk score
        med.risk_score = med.calculate_risk_score()
        
        db.session.add(med)
        db.session.commit()
        inventory_index.add_medicine(med)
        flash('Medicine added successfully', 'success')
        return redirect(url_for('main.medicines'))
    
    return render_template('add.html')


@main_bp.route('/delete/<int:med_id>', methods=['POST'])
@login_required
def delete_medicine(med_id: int):
    med = Medicine.query.get_or_404(med_id)
    if med.user_id != int(current_user.get_id()):
        flash('Not authorized to delete this item', 'danger')
        return redirect(url_for('main.medicines'))
    inventory_index.remove_medicine(med)
    db.session.delete(med)
    db.session.commit()
    flash('Medicine deleted', 'success')
    return redirect(url_for('main.medicines'))


@main_bp.route('/search')
@login_required
def search():
    q = request.args.get('q', '').strip()
    results = inventory_index.search_by_name(q) if q else []
    return render_template('search.html', q=q, results=results)


# Advanced Analytics Routes
@main_bp.route('/analytics')
@login_required
def analytics():
    """Advanced analytics dashboard"""
    user_id = int(current_user.get_id())
    advanced_service = get_advanced_service(user_id)
    
    # Get comprehensive analytics
    analytics_data = advanced_service.get_dashboard_analytics()
    if not isinstance(analytics_data, dict):
        analytics_data = _to_plain_dict(analytics_data)
    analytics_data = _ensure_analytics_defaults(analytics_data)
    
    # Get consumption analytics
    consumption_analytics = advanced_service.get_consumption_analytics(30)
    
    return render_template('analytics.html', 
                         analytics=analytics_data,
                         consumption=consumption_analytics)


@main_bp.route('/analytics/api/consumption', methods=['POST'])
@login_required
def record_consumption():
    """Record medicine consumption"""
    user_id = int(current_user.get_id())
    advanced_service = get_advanced_service(user_id)
    
    data = request.get_json()
    medicine_id = data.get('medicine_id')
    quantity = data.get('quantity', 0)
    notes = data.get('notes', '')
    
    if not medicine_id or quantity <= 0:
        return jsonify({'error': 'Invalid data'}), 400
    
    try:
        record = advanced_service.create_consumption_record(medicine_id, quantity, notes)
        return jsonify({
            'success': True,
            'record_id': record.id,
            'message': 'Consumption recorded successfully'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main_bp.route('/analytics/api/recommendations')
@login_required
def get_recommendations():
    """Get smart recommendations"""
    user_id = int(current_user.get_id())
    advanced_service = get_advanced_service(user_id)
    
    recommendations = advanced_service.generate_smart_recommendations()
    return jsonify(recommendations)


@main_bp.route('/analytics/api/abc-analysis')
@login_required
def abc_analysis():
    """Get ABC analysis results"""
    user_id = int(current_user.get_id())
    advanced_service = get_advanced_service(user_id)
    
    abc_data = advanced_service.analytics.calculate_abc_analysis()
    return jsonify(abc_data)


@main_bp.route('/analytics/api/xyz-analysis')
@login_required
def xyz_analysis():
    """Get XYZ analysis results"""
    user_id = int(current_user.get_id())
    advanced_service = get_advanced_service(user_id)
    
    xyz_data = advanced_service.analytics.calculate_xyz_analysis()
    return jsonify(xyz_data)


@main_bp.route('/analytics/api/consumption-trends')
@login_required
def consumption_trends():
    """Get consumption trends data"""
    user_id = int(current_user.get_id())
    days = request.args.get('days', 30, type=int)
    
    advanced_service = get_advanced_service(user_id)
    consumption_data = advanced_service.get_consumption_analytics(days)
    
    return jsonify(consumption_data)


@main_bp.route('/alerts')
@login_required
def alerts():
    """View all alerts"""
    user_id = int(current_user.get_id())
    
    # Generate new alerts
    advanced_service = get_advanced_service(user_id)
    advanced_service.generate_alerts()
    
    # Get all alerts
    all_alerts = InventoryAlert.query.filter(
        InventoryAlert.user_id == user_id
    ).order_by(InventoryAlert.created_at.desc()).all()
    
    return render_template('alerts.html', alerts=all_alerts)


@main_bp.route('/alerts/<int:alert_id>/mark-read', methods=['POST'])
@login_required
def mark_alert_read(alert_id):
    """Mark alert as read"""
    user_id = int(current_user.get_id())
    
    alert = InventoryAlert.query.filter(
        InventoryAlert.id == alert_id,
        InventoryAlert.user_id == user_id
    ).first()
    
    if alert:
        alert.is_read = True
        db.session.commit()
        return jsonify({'success': True})
    
    return jsonify({'error': 'Alert not found'}), 404


@main_bp.route('/medicines/<int:med_id>/consume', methods=['POST'])
@login_required
def consume_medicine(med_id):
    """Record medicine consumption"""
    user_id = int(current_user.get_id())
    
    # Check if medicine belongs to user
    medicine = Medicine.query.filter(
        Medicine.id == med_id,
        Medicine.user_id == user_id
    ).first()
    
    if not medicine:
        return jsonify({'error': 'Medicine not found'}), 404
    
    data = request.get_json()
    quantity = data.get('quantity', 0)
    notes = data.get('notes', '')
    
    if quantity <= 0 or quantity > medicine.quantity:
        return jsonify({'error': 'Invalid quantity'}), 400
    
    try:
        advanced_service = get_advanced_service(user_id)
        record = advanced_service.create_consumption_record(med_id, quantity, notes)
        
        return jsonify({
            'success': True,
            'remaining_quantity': medicine.quantity,
            'message': f'Consumed {quantity} units of {medicine.name}'
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main_bp.route('/reports')
@login_required
def reports():
    """Generate various reports"""
    user_id = int(current_user.get_id())
    advanced_service = get_advanced_service(user_id)
    
    # Get analytics data for reports
    analytics_data = advanced_service.get_dashboard_analytics()
    if not isinstance(analytics_data, dict):
        analytics_data = _to_plain_dict(analytics_data)
    analytics_data = _ensure_analytics_defaults(analytics_data)

    return render_template('reports.html', analytics=analytics_data)


@main_bp.route('/reports/api/export')
@login_required
def export_data():
    """Export inventory data as CSV"""
    user_id = int(current_user.get_id())
    
    medicines = Medicine.query.filter(Medicine.user_id == user_id).all()
    
    # Convert to CSV format
    csv_data = []
    for med in medicines:
        csv_data.append({
            'Name': med.name,
            'Category': med.category or '',
            'Manufacturer': med.manufacturer or '',
            'Quantity': med.quantity,
            'Expiry Date': med.expiry_date.strftime('%Y-%m-%d'),
            'Purchase Price': med.purchase_price or 0,
            'Selling Price': med.selling_price or 0,
            'Risk Score': round(med.calculate_risk_score(), 2),
            'Consumption Rate': med.consumption_rate,
            'Days Until Expiry': med.days_until_expiry(),
            'Is Low Stock': med.is_low_stock(),
            'Is Overstocked': med.is_overstocked()
        })
    
    return jsonify(csv_data)


# Machine Learning Routes
@main_bp.route('/ml/train', methods=['POST'])
@login_required
def train_ml_models():
    """Train all ML models"""
    user_id = int(current_user.get_id())
    ml_manager = MLModelManager(user_id)
    
    try:
        results = ml_manager.train_all_models()
        return jsonify({
            'success': True,
            'results': results,
            'message': 'ML models training completed'
        })
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500


@main_bp.route('/ml/insights')
@login_required
def get_ml_insights():
    """Get ML model insights"""
    user_id = int(current_user.get_id())
    ml_manager = MLModelManager(user_id)
    
    try:
        insights = ml_manager.get_ml_insights()
        return jsonify(insights)
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main_bp.route('/ml/forecast/<int:medicine_id>')
@login_required
def get_demand_forecast(medicine_id):
    """Get demand forecast for a specific medicine"""
    user_id = int(current_user.get_id())
    ml_manager = MLModelManager(user_id)
    
    try:
        days_ahead = request.args.get('days', 7, type=int)
        forecast = ml_manager.demand_model.predict_demand(medicine_id, days_ahead)
        
        return jsonify({
            'medicine_id': medicine_id,
            'forecast': forecast,
            'days_ahead': days_ahead
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main_bp.route('/ml/anomalies')
@login_required
def get_anomalies():
    """Get detected anomalies"""
    user_id = int(current_user.get_id())
    ml_manager = MLModelManager(user_id)
    
    try:
        days = request.args.get('days', 7, type=int)
        anomalies = ml_manager.anomaly_model.detect_anomalies(days)
        
        return jsonify({
            'anomalies': anomalies,
            'count': len(anomalies),
            'days_analyzed': days
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


@main_bp.route('/ml/clusters')
@login_required
def get_clusters():
    """Get medicine clusters"""
    user_id = int(current_user.get_id())
    ml_manager = MLModelManager(user_id)
    
    try:
        clusters = ml_manager.clustering_model.predict_clusters()
        
        return jsonify({
            'clusters': clusters,
            'total_medicines': len(clusters),
            'unique_clusters': len(set(c['cluster'] for c in clusters))
        })
    except Exception as e:
        return jsonify({'error': str(e)}), 500


