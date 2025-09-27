from flask import Blueprint, render_template, session, redirect, url_for, request, flash
from flask_login import login_required, current_user
from ..models import Medicine, ConsumptionRecord
from .. import db
from datetime import date

shop_bp = Blueprint('shop', __name__, url_prefix='/shop')


def _get_cart():
    return session.setdefault('cart', {})


@shop_bp.route('/')
def catalog():
    # show all available medicines with quantity > 0
    meds = Medicine.query.filter(Medicine.quantity > 0).all()
    return render_template('shop/catalog.html', medicines=meds)


@shop_bp.route('/product/<int:med_id>')
def product(med_id):
    med = Medicine.query.get_or_404(med_id)
    return render_template('shop/product.html', medicine=med)


@shop_bp.route('/cart')
def cart():
    cart = _get_cart()
    items = []
    total = 0
    for mid, qty in cart.items():
        med = Medicine.query.get(int(mid))
        if not med:
            continue
        items.append({'medicine': med, 'quantity': qty})
        total += (med.selling_price or 0) * qty
    return render_template('shop/cart.html', items=items, total=total)


@shop_bp.route('/cart/add', methods=['POST'])
def add_to_cart():
    med_id = request.form.get('med_id')
    qty = int(request.form.get('quantity', '1') or 1)
    med = Medicine.query.get_or_404(int(med_id))
    if med.quantity < qty:
        flash('Not enough stock available', 'danger')
        return redirect(url_for('shop.product', med_id=med.id))
    cart = _get_cart()
    cart[str(med.id)] = cart.get(str(med.id), 0) + qty
    session.modified = True
    flash('Added to cart', 'success')
    return redirect(url_for('shop.cart'))


@shop_bp.route('/checkout', methods=['POST'])
@login_required
def checkout():
    cart = _get_cart()
    if not cart:
        flash('Cart is empty', 'warning')
        return redirect(url_for('shop.cart'))

    user_id = int(current_user.get_id())
    created = 0
    for mid, qty in list(cart.items()):
        med = Medicine.query.get(int(mid))
        if not med:
            continue
        # ensure stock
        actual_qty = min(qty, med.quantity)
        if actual_qty <= 0:
            continue
        # create consumption record
        rec = ConsumptionRecord(
            medicine_id=med.id,
            user_id=user_id,
            quantity_consumed=actual_qty,
            consumption_date=date.today(),
        )
        db.session.add(rec)
        med.quantity = med.quantity - actual_qty
        created += 1
        # remove from cart
        cart.pop(mid, None)

    try:
        db.session.commit()
        session.modified = True
        flash(f'Checkout complete ({created} items)', 'success')
    except Exception as e:
        db.session.rollback()
        flash(f'Checkout failed: {e}', 'danger')

    return redirect(url_for('shop.catalog'))
