from flask import Flask
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager
from pathlib import Path
import os
from dotenv import load_dotenv

db = SQLAlchemy()
login_manager = LoginManager()


def create_app() -> Flask:
    # Load environment variables from .env if present
    load_dotenv()

    app = Flask(__name__, instance_relative_config=True)

    # Basic config
    app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'dev-secret-key')
    db_path = os.path.join(app.instance_path, 'app.sqlite')
    Path(app.instance_path).mkdir(parents=True, exist_ok=True)
    app.config['SQLALCHEMY_DATABASE_URI'] = f'sqlite:///{db_path}'
    app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False

    # Init extensions
    db.init_app(app)
    login_manager.init_app(app)
    login_manager.login_view = 'auth.login'

    # Blueprints
    from .auth.routes import auth_bp
    from .main.routes import main_bp
    app.register_blueprint(auth_bp)
    app.register_blueprint(main_bp)

    # Create tables
    with app.app_context():
        from .models import User, Medicine, ConsumptionRecord, AnalyticsCache, InventoryAlert
        db.create_all()
        # Ensure new columns exist for SQLite (simple migration)
        try:
            from sqlalchemy import text
            cols = [row[1] for row in db.session.execute(text("PRAGMA table_info(medicine)")).fetchall()]
            if 'user_id' not in cols:
                db.session.execute(text("ALTER TABLE medicine ADD COLUMN user_id INTEGER"))
            if 'category' not in cols:
                db.session.execute(text("ALTER TABLE medicine ADD COLUMN category VARCHAR(50)"))
            if 'manufacturer' not in cols:
                db.session.execute(text("ALTER TABLE medicine ADD COLUMN manufacturer VARCHAR(100)"))
            if 'batch_number' not in cols:
                db.session.execute(text("ALTER TABLE medicine ADD COLUMN batch_number VARCHAR(50)"))
            if 'purchase_price' not in cols:
                db.session.execute(text("ALTER TABLE medicine ADD COLUMN purchase_price FLOAT"))
            if 'selling_price' not in cols:
                db.session.execute(text("ALTER TABLE medicine ADD COLUMN selling_price FLOAT"))
            if 'min_stock_level' not in cols:
                db.session.execute(text("ALTER TABLE medicine ADD COLUMN min_stock_level INTEGER DEFAULT 10"))
            if 'max_stock_level' not in cols:
                db.session.execute(text("ALTER TABLE medicine ADD COLUMN max_stock_level INTEGER DEFAULT 1000"))
            if 'created_at' not in cols:
                db.session.execute(text("ALTER TABLE medicine ADD COLUMN created_at DATETIME"))
            if 'updated_at' not in cols:
                db.session.execute(text("ALTER TABLE medicine ADD COLUMN updated_at DATETIME"))
            if 'consumption_rate' not in cols:
                db.session.execute(text("ALTER TABLE medicine ADD COLUMN consumption_rate FLOAT DEFAULT 0.0"))
            if 'demand_forecast' not in cols:
                db.session.execute(text("ALTER TABLE medicine ADD COLUMN demand_forecast FLOAT DEFAULT 0.0"))
            if 'risk_score' not in cols:
                db.session.execute(text("ALTER TABLE medicine ADD COLUMN risk_score FLOAT DEFAULT 0.0"))
            if 'last_consumption_date' not in cols:
                db.session.execute(text("ALTER TABLE medicine ADD COLUMN last_consumption_date DATE"))
            db.session.commit()
            
            ucols = [row[1] for row in db.session.execute(text("PRAGMA table_info(user)")).fetchall()]
            if 'full_name' not in ucols:
                db.session.execute(text("ALTER TABLE user ADD COLUMN full_name VARCHAR(120)"))
            if 'email' not in ucols:
                db.session.execute(text("ALTER TABLE user ADD COLUMN email VARCHAR(120)"))
            if 'phone' not in ucols:
                db.session.execute(text("ALTER TABLE user ADD COLUMN phone VARCHAR(30)"))
            if 'profile_pic' not in ucols:
                db.session.execute(text("ALTER TABLE user ADD COLUMN profile_pic VARCHAR(255)"))
            db.session.commit()
        except Exception as e:
            print(f"Migration error: {e}")
            pass

    return app


