# Project_DS — Medicine Inventory & Analytics

This is a Flask-based medicine inventory web application with analytics and simple ML-ready components. It uses Flask, SQLAlchemy and a small in-memory index for fast expiry/search operations. This README explains how to run locally and how to deploy quickly to a service such as Render (no external DB required for a demo).

## Quick local run (Windows / PowerShell)

1. Create and activate virtualenv:

```powershell
cd C:\Users\91920\OneDrive\Desktop\Project_DS
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2. Install dependencies:

```powershell
pip install -r requirements.txt
```

3. Run the app (development):

```powershell
# from project root
python app.py
```

Open http://127.0.0.1:5000 in your browser.

## Prepare for deployment (Render / Heroku style)

Files already added to this repo for deployment:
- `Procfile` — start command for gunicorn
- `requirements.txt` — includes `gunicorn`

Recommended quick deploy (Render):
1. Push your repository to GitHub (see commands below).
2. Create a new Web Service on Render and connect your GitHub repo.
3. Use these build/start commands in the Render UI (or let `Procfile` be used):

Build command:
```
pip install -r requirements.txt
```

Start command:
```
gunicorn -b 0.0.0.0:$PORT app:app
```

Render will provide an HTTPS URL once the service is created and built.

## Git commands (push to GitHub)

Replace `YOUR_USERNAME` and `YOUR_REPO` with your GitHub username/repo name.

```powershell
# from project root
git init
git add -A
git commit -m "Prepare project for deployment"
git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
git branch -M main
git push -u origin main
```

If you need to change the remote URL:

```powershell
git remote set-url origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
```

## Notes and caveats
- The app uses a local SQLite DB (`instance/app.sqlite`) by default. This is fine for quick demos but is ephemeral on many hosts. For persistence use a hosted database (Postgres) and set its URI in your environment.
- Set `SECRET_KEY` in the Render (or host) environment variables for production.
- Use `DEBUG=False` in production.

## Want help?
If you want, I can:
- create a GitHub Actions workflow to auto-deploy to Render on push,
- modify the config to use `DATABASE_URL` environment variable (for Postgres), or
- walk you step-by-step while you push and deploy.

Tell me which you prefer and I will proceed.
# MediTrack - Medicine Expiry & Stock Management

A Flask app using Min-Heap and HashMap to prioritize near-expiry medicines and enable instant name lookup.

## Quickstart (Windows PowerShell)
```bash
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py   # creates DB
python seed.py  # creates admin and seeds sample data
python app.py
```
Login at `http://localhost:5000` with `admin` / `admin123`.

- Data structures: `app/ds/structures.py`
- Routes/UI: `app/main/routes.py`, templates under `app/templates/`
- Optional CSV seed: put `data/medicines.csv` (name,quantity,expiry_date YYYY-MM-DD) then run `python seed.py`.
