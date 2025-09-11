from datetime import date, datetime
import csv
from pathlib import Path
from app import create_app, db
from app.models import User, Medicine

app = create_app()

with app.app_context():
    admin = User.query.filter_by(username='admin').first()
    if not admin:
        admin = User(username='admin', is_admin=True)
        admin.set_password('admin123')
        db.session.add(admin)
    # Seed from CSV if present, else add a few defaults
    if Medicine.query.filter(Medicine.user_id == admin.id).count() == 0:
        csv_path = Path('data/medicines.csv')
        meds = []
        if csv_path.exists():
            with csv_path.open(newline='', encoding='utf-8') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    try:
                        name = row['name'].strip()
                        quantity = int(row['quantity'])
                        expiry = datetime.strptime(row['expiry_date'].strip(), '%Y-%m-%d').date()
                        meds.append(Medicine(name=name, quantity=quantity, expiry_date=expiry, user_id=admin.id))
                    except Exception as e:
                        print('Skipping row due to error:', row, e)
        else:
            meds = [
                Medicine(name='Paracetamol', quantity=120, expiry_date=date(2025, 10, 10), user_id=admin.id),
                Medicine(name='Amoxicillin', quantity=60, expiry_date=date(2025, 9, 15), user_id=admin.id),
                Medicine(name='Cough Syrup', quantity=30, expiry_date=date(2025, 9, 5), user_id=admin.id),
            ]
        if meds:
            db.session.add_all(meds)
    db.session.commit()
    print('Seed complete. Login with admin / admin123')


