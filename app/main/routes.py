from datetime import datetime, date, timedelta
from flask import Blueprint, render_template, redirect, url_for, request, flash, jsonify, send_file
from flask_login import login_required, current_user
from ..models import Medicine, User, ConsumptionRecord, InventoryAlert
from .. import db
from ..services import inventory_index, cleanup_expired, get_advanced_service
from ..ds.ml_models import MLModelManager
from sqlalchemy import or_, func
from io import BytesIO
import json
import io
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
@main_bp.route('/sales-chart')
@login_required
def sales_chart():
    """Simple sales chart showing medicine sales by month"""
    user_id = int(current_user.get_id())
    
    # Get consumption data for the last 12 months
    from datetime import datetime, timedelta
    from sqlalchemy import func, extract
    
    # Calculate date range for last 12 months
    end_date = datetime.now().date()
    start_date = end_date - timedelta(days=365)
    
    # Query consumption data grouped by month
    monthly_sales = db.session.query(
        extract('year', ConsumptionRecord.consumption_date).label('year'),
        extract('month', ConsumptionRecord.consumption_date).label('month'),
        func.sum(ConsumptionRecord.quantity_consumed).label('total_quantity'),
        func.count(ConsumptionRecord.id).label('total_transactions')
    ).join(Medicine).filter(
        ConsumptionRecord.user_id == user_id,
        ConsumptionRecord.consumption_date >= start_date,
        ConsumptionRecord.consumption_date <= end_date
    ).group_by(
        extract('year', ConsumptionRecord.consumption_date),
        extract('month', ConsumptionRecord.consumption_date)
    ).order_by('year', 'month').all()
    
    # Format data for chart
    chart_data = {
        'labels': [],
        'quantities': [],
        'transactions': []
    }
    
    # Create month labels and data
    current_date = start_date.replace(day=1)
    while current_date <= end_date:
        month_label = current_date.strftime('%b %Y')
        chart_data['labels'].append(month_label)
        
        # Find data for this month
        month_data = next(
            (row for row in monthly_sales 
             if row.year == current_date.year and row.month == current_date.month), 
            None
        )
        
        if month_data:
            chart_data['quantities'].append(int(month_data.total_quantity))
            chart_data['transactions'].append(int(month_data.total_transactions))
        else:
            chart_data['quantities'].append(0)
            chart_data['transactions'].append(0)
        
        # Move to next month
        if current_date.month == 12:
            current_date = current_date.replace(year=current_date.year + 1, month=1)
        else:
            current_date = current_date.replace(month=current_date.month + 1)
    
    return render_template('sales_chart.html', chart_data=chart_data)


# Simple consumption recording for sales chart
@main_bp.route('/api/record-consumption', methods=['POST'])
@login_required
def record_consumption():
    """Record medicine consumption for sales tracking"""
    user_id = int(current_user.get_id())
    
    data = request.get_json()
    medicine_id = data.get('medicine_id')
    quantity = data.get('quantity', 0)
    notes = data.get('notes', '')
    
    if not medicine_id or quantity <= 0:
        return jsonify({'error': 'Invalid data'}), 400
    
    try:
        # Check if medicine belongs to user
        medicine = Medicine.query.filter(
            Medicine.id == medicine_id,
            Medicine.user_id == user_id
        ).first()
        
        if not medicine:
            return jsonify({'error': 'Medicine not found'}), 404
        
        # Create consumption record
        record = ConsumptionRecord(
            medicine_id=medicine_id,
            user_id=user_id,
            quantity_consumed=quantity,
            consumption_date=date.today(),
            notes=notes
        )
        
        # Update medicine quantity
        medicine.quantity = max(0, medicine.quantity - quantity)
        medicine.last_consumption_date = date.today()
        
        db.session.add(record)
        db.session.commit()
        
        return jsonify({
            'success': True,
            'record_id': record.id,
            'remaining_quantity': medicine.quantity,
            'message': f'Consumed {quantity} units of {medicine.name}'
        })
    except Exception as e:
        db.session.rollback()
        return jsonify({'error': str(e)}), 500


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


@main_bp.route('/alerts/mark-all', methods=['POST'])
@login_required
def mark_all_alerts():
    """Mark all unread alerts as read for the current user."""
    user_id = int(current_user.get_id())
    try:
        updated = InventoryAlert.query.filter(
            InventoryAlert.user_id == user_id,
            InventoryAlert.is_read == False
        ).update({InventoryAlert.is_read: True}, synchronize_session=False)
        db.session.commit()
        return jsonify({'success': True, 'updated': int(updated)})
    except Exception as e:
        db.session.rollback()
        return jsonify({'success': False, 'error': str(e)}), 500


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


# -------------------------
# Reports: JSON Endpoints
# -------------------------

@main_bp.route('/reports/api/inventory-summary')
@login_required
def api_inventory_summary():
    user_id = int(current_user.get_id())
    from ..models import Medicine
    meds = Medicine.query.filter(Medicine.user_id == user_id).all()
    total_medicines = len(meds)
    expired = sum(1 for m in meds if m.is_expired())
    expiring_soon = sum(1 for m in meds if 0 < m.days_until_expiry() <= 30)
    low_stock = sum(1 for m in meds if m.is_low_stock())
    overstocked = sum(1 for m in meds if m.is_overstocked())
    total_value = sum(m.quantity * (m.selling_price or 0) for m in meds)
    categories = {}
    for m in meds:
        key = m.category or 'Uncategorized'
        categories[key] = categories.get(key, 0) + 1
    data = {
        'generated_at': datetime.utcnow().isoformat(),
        'total_medicines': total_medicines,
        'expired_count': expired,
        'expiring_soon_count': expiring_soon,
        'low_stock_count': low_stock,
        'overstocked_count': overstocked,
        'total_inventory_value': total_value,
        'categories': categories,
    }
    return jsonify(data)


@main_bp.route('/reports/api/consumption-analysis')
@login_required
def api_consumption_analysis():
    user_id = int(current_user.get_id())
    # Last 30 days trends
    start_date = date.today() - timedelta(days=29)
    rows = (
        db.session.query(
            ConsumptionRecord.consumption_date,
            func.sum(ConsumptionRecord.quantity_consumed)
        )
        .filter(
            ConsumptionRecord.user_id == user_id,
            ConsumptionRecord.consumption_date >= start_date
        )
        .group_by(ConsumptionRecord.consumption_date)
        .order_by(ConsumptionRecord.consumption_date)
        .all()
    )
    trends = {r[0].isoformat(): int(r[1]) for r in rows}
    total_consumption = int(sum(r[1] for r in rows))
    days = max(1, (date.today() - start_date).days + 1)
    daily_average = total_consumption / days
    data = {
        'generated_at': datetime.utcnow().isoformat(),
        'window_days': days,
        'total_consumption': total_consumption,
        'daily_average': round(daily_average, 2),
        'daily_trends': trends,
    }
    return jsonify(data)


@main_bp.route('/reports/api/expiry-risk')
@login_required
def api_expiry_risk():
    user_id = int(current_user.get_id())
    meds = Medicine.query.filter(Medicine.user_id == user_id).all()
    expiring_soon = [m for m in meds if 0 < m.days_until_expiry() <= 30]
    at_risk_value = sum(m.quantity * (m.selling_price or 0) for m in expiring_soon)
    data = {
        'generated_at': datetime.utcnow().isoformat(),
        'expired_count': len([m for m in meds if m.is_expired()]),
        'expiring_soon_count': len(expiring_soon),
        'total_at_risk_value': at_risk_value,
        'items': [
            {
                'id': m.id,
                'name': m.name,
                'days_until_expiry': m.days_until_expiry(),
                'quantity': m.quantity,
                'value': m.quantity * (m.selling_price or 0),
            }
            for m in expiring_soon
        ],
    }
    return jsonify(data)


@main_bp.route('/reports/api/financial')
@login_required
def api_financial():
    user_id = int(current_user.get_id())
    meds = Medicine.query.filter(Medicine.user_id == user_id).all()
    total_value = sum(m.quantity * (m.selling_price or 0) for m in meds)
    low_stock = [m for m in meds if m.is_low_stock()]
    overstocked = [m for m in meds if m.is_overstocked()]
    low_stock_value = sum(m.quantity * (m.selling_price or 0) for m in low_stock)
    overstocked_value = sum(m.quantity * (m.selling_price or 0) for m in overstocked)
    data = {
        'generated_at': datetime.utcnow().isoformat(),
        'total_inventory_value': total_value,
        'low_stock_value': low_stock_value,
        'overstocked_value': overstocked_value,
    }
    return jsonify(data)


# -------------------------
# Reports: PDF Endpoints
# -------------------------

def _pdf_headers(filename: str) -> dict:
    return {
        'Content-Type': 'application/pdf',
        'Content-Disposition': f'attachment; filename="{filename}"'
    }


def _render_simple_pdf(title: str, lines: list[str]) -> BytesIO:
    from reportlab.pdfgen import canvas  # type: ignore
    from reportlab.lib.pagesizes import A4  # type: ignore
    buffer = BytesIO()
    c = canvas.Canvas(buffer, pagesize=A4)
    width, height = A4
    y = height - 50
    c.setFont("Helvetica-Bold", 16)
    c.drawString(50, y, title)
    y -= 20
    c.setFont("Helvetica", 11)
    for line in lines:
        if y < 50:
            c.showPage()
            y = height - 50
            c.setFont("Helvetica", 11)
        c.drawString(50, y, str(line))
        y -= 16
    c.showPage()
    c.save()
    buffer.seek(0)
    return buffer


@main_bp.route('/reports/api/inventory-summary.pdf')
@login_required
def api_inventory_summary_pdf():
    resp = api_inventory_summary().json
    lines = [
        f"Generated: {resp.get('generated_at')}",
        f"Total Items: {resp.get('total_medicines')}",
        f"Expired: {resp.get('expired_count')}",
        f"Expiring Soon: {resp.get('expiring_soon_count')}",
        f"Low Stock: {resp.get('low_stock_count')}",
        f"Overstocked: {resp.get('overstocked_count')}",
        f"Total Inventory Value: {resp.get('total_inventory_value')}",
        "Categories:",
    ]
    for k, v in (resp.get('categories') or {}).items():
        lines.append(f"  - {k}: {v}")
    pdf = _render_simple_pdf("Inventory Summary", lines)
    return send_file(pdf, mimetype='application/pdf', as_attachment=True, download_name='inventory_summary.pdf')


@main_bp.route('/reports/api/consumption-analysis.pdf')
@login_required
def api_consumption_analysis_pdf():
    resp = api_consumption_analysis().json
    lines = [
        f"Generated: {resp.get('generated_at')}",
        f"Window Days: {resp.get('window_days')}",
        f"Total Consumption: {resp.get('total_consumption')}",
        f"Daily Average: {resp.get('daily_average')}",
        "Daily Trends:",
    ]
    for d, qty in (resp.get('daily_trends') or {}).items():
        lines.append(f"  - {d}: {qty}")
    pdf = _render_simple_pdf("Consumption Analysis", lines)
    return send_file(pdf, mimetype='application/pdf', as_attachment=True, download_name='consumption_analysis.pdf')


@main_bp.route('/reports/api/expiry-risk.pdf')
@login_required
def api_expiry_risk_pdf():
    resp = api_expiry_risk().json
    lines = [
        f"Generated: {resp.get('generated_at')}",
        f"Expired: {resp.get('expired_count')}",
        f"Expiring Soon: {resp.get('expiring_soon_count')}",
        f"Total At-Risk Value: {resp.get('total_at_risk_value')}",
        "Items:",
    ]
    for it in (resp.get('items') or []):
        lines.append(f"  - {it.get('name')} (ID {it.get('id')}): {it.get('days_until_expiry')} days, qty {it.get('quantity')}, value {it.get('value')}")
    pdf = _render_simple_pdf("Expiry Risk", lines)
    return send_file(pdf, mimetype='application/pdf', as_attachment=True, download_name='expiry_risk.pdf')


@main_bp.route('/reports/api/financial.pdf')
@login_required
def api_financial_pdf():
    resp = api_financial().json
    lines = [
        f"Generated: {resp.get('generated_at')}",
        f"Total Inventory Value: {resp.get('total_inventory_value')}",
        f"Low Stock Value: {resp.get('low_stock_value')}",
        f"Overstocked Value: {resp.get('overstocked_value')}",
    ]
    pdf = _render_simple_pdf("Financial Report", lines)
    return send_file(pdf, mimetype='application/pdf', as_attachment=True, download_name='financial_report.pdf')


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


@main_bp.route('/upload-csv', methods=['GET', 'POST'])
@login_required
def upload_csv():
    """Handle CSV file upload for medicine data"""
    if request.method == 'GET':
        return render_template('upload_csv.html')
    
    user_id = int(current_user.get_id())
    
    # Check if file was uploaded
    if 'csv_file' not in request.files:
        flash('No file selected', 'danger')
        return redirect(url_for('main.upload_csv'))
    
    file = request.files['csv_file']
    
    if file.filename == '':
        flash('No file selected', 'danger')
        return redirect(url_for('main.upload_csv'))
    
    if not file.filename.lower().endswith('.csv'):
        flash('Please upload a CSV file', 'danger')
        return redirect(url_for('main.upload_csv'))
    
    try:
        # Read CSV file
        import pandas as pd  # local import to avoid module-level dependency issues

        csv_data = file.read().decode('utf-8')
        df = pd.read_csv(io.StringIO(csv_data))
        
        # Validate required columns
        required_columns = ['name', 'quantity', 'expiry_date']
        missing_columns = [col for col in required_columns if col not in df.columns]
        
        if missing_columns:
            flash(f'Missing required columns: {", ".join(missing_columns)}', 'danger')
            return redirect(url_for('main.upload_csv'))
        
        # Process each row
        success_count = 0
        error_count = 0
        errors = []
        
        for index, row in df.iterrows():
            try:
                # Extract data from row
                name = str(row['name']).strip()
                quantity = int(row['quantity'])
                expiry_str = str(row['expiry_date']).strip()
                
                # Optional fields with defaults
                category = str(row.get('category', '')).strip() or None
                manufacturer = str(row.get('manufacturer', '')).strip() or None
                batch_number = str(row.get('batch_number', '')).strip() or None
                purchase_price = float(row.get('purchase_price', 0)) or None
                selling_price = float(row.get('selling_price', 0)) or None
                min_stock_level = int(row.get('min_stock_level', 10))
                max_stock_level = int(row.get('max_stock_level', 1000))
                
                # Validate data
                if not name or quantity <= 0:
                    errors.append(f'Row {index + 1}: Invalid name or quantity')
                    error_count += 1
                    continue
                
                # Parse expiry date (support common formats with '-' or '/')
                parsed = None
                for fmt in (
                    '%Y-%m-%d',  # 2025-12-31
                    '%d-%m-%Y',  # 31-12-2025
                    '%m-%d-%Y',  # 12-31-2025
                    '%d/%m/%Y',  # 31/12/2025
                    '%m/%d/%Y',  # 12/31/2025
                ):
                    try:
                        parsed = datetime.strptime(expiry_str, fmt).date()
                        break
                    except ValueError:
                        continue
                if not parsed:
                    errors.append(f'Row {index + 1}: Invalid date format for {expiry_str}')
                    error_count += 1
                    continue
                expiry_date = parsed
                
                # Check if medicine already exists
                existing = Medicine.query.filter(
                    Medicine.name == name,
                    Medicine.user_id == user_id
                ).first()
                
                if existing:
                    # Update existing medicine (persist first, rebuild index later)
                    existing.quantity += quantity
                    existing.category = category or existing.category
                    existing.manufacturer = manufacturer or existing.manufacturer
                    existing.batch_number = batch_number or existing.batch_number
                    existing.purchase_price = purchase_price or existing.purchase_price
                    existing.selling_price = selling_price or existing.selling_price
                    existing.min_stock_level = min_stock_level
                    existing.max_stock_level = max_stock_level
                    existing.risk_score = existing.calculate_risk_score()
                else:
                    # Create new medicine
                    medicine = Medicine(
                        name=name,
                        quantity=quantity,
                        expiry_date=expiry_date,
                        user_id=user_id,
                        category=category,
                        manufacturer=manufacturer,
                        batch_number=batch_number,
                        purchase_price=purchase_price,
                        selling_price=selling_price,
                        min_stock_level=min_stock_level,
                        max_stock_level=max_stock_level
                    )
                    medicine.risk_score = medicine.calculate_risk_score()
                    db.session.add(medicine)
                
                success_count += 1
                
            except Exception as e:
                errors.append(f'Row {index + 1}: {str(e)}')
                error_count += 1
        
        # Commit all changes first so new medicines have IDs
        db.session.commit()

        # Rebuild inventory index for this user so uploaded/updated medicines are available
        try:
            inventory_index.rebuild(user_id=user_id)
        except Exception:
            # Non-fatal - index rebuild should not break the upload flow
            pass
        
        # Show results
        if success_count > 0:
            flash(f'Successfully processed {success_count} medicines', 'success')
        
        if error_count > 0:
            error_msg = f'Failed to process {error_count} medicines. '
            if errors:
                error_msg += f'Errors: {"; ".join(errors[:5])}'
                if len(errors) > 5:
                    error_msg += f'... and {len(errors) - 5} more errors'
            flash(error_msg, 'warning')
        
        return redirect(url_for('main.medicines'))
        
    except Exception as e:
        flash(f'Error processing CSV file: {str(e)}', 'danger')
        return redirect(url_for('main.upload_csv'))


