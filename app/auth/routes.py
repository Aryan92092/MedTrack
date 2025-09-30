from flask import Blueprint, render_template, redirect, url_for, request, flash
from flask_login import login_user, logout_user, login_required
from ..models import User
from .. import db


auth_bp = Blueprint('auth', __name__, url_prefix='/auth')


@auth_bp.route('/login', methods=['GET', 'POST'])
def login():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        # Enforce minimum password length for security
        if len(password) < 8:
            flash('Password must be at least 8 characters long', 'danger')
            return render_template('login.html')
        user = User.query.filter_by(username=username).first()
        if not user or not user.check_password(password):
            flash('Invalid credentials', 'danger')
            return render_template('login.html')
        login_user(user)
        return redirect(url_for('main.dashboard'))
    return render_template('login.html')


@auth_bp.route('/logout')
@login_required
def logout():
    logout_user()
    return redirect(url_for('auth.login'))


@auth_bp.route('/register', methods=['GET', 'POST'])
def register():
    if request.method == 'POST':
        username = request.form.get('username', '').strip()
        password = request.form.get('password', '')
        confirm = request.form.get('confirm', '')
        if not username or not password:
            flash('Username and password are required', 'danger')
            return render_template('register.html')
        if len(password) < 8:
            flash('Password must be at least 8 characters long', 'danger')
            return render_template('register.html')
        if password != confirm:
            flash('Passwords do not match', 'danger')
            return render_template('register.html')
        existing = User.query.filter_by(username=username).first()
        if existing:
            flash('Username is already taken', 'danger')
            return render_template('register.html')
        user = User(username=username, is_admin=False)
        user.set_password(password)
        db.session.add(user)
        db.session.commit()
        flash('Registration successful. Please log in.', 'success')
        return redirect(url_for('auth.login'))
    return render_template('register.html')


