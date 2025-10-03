# MediTrack - Medicine Expiry & Stock Management

A Flask app using Min-Heap and HashMap to prioritize near-expiry medicines and enable instant name lookup.

## Quickstart (Windows PowerShell)

1.  **Set up the environment** (only needs to be done once):
    ```powershell
    # Navigate to the project directory
    cd C:\Users\91920\OneDrive\Desktop\Project_DS

    # Create and activate a virtual environment
    python -m venv venv
    .\venv\Scripts\Activate.ps1

    # Install required packages
    pip install -r requirements.txt
    ```

2.  **Run the application**:
    ```powershell
    # Make sure your virtual environment is active. 
    # Your command prompt should start with (venv).
    # Then, run the application using the run.py script.
    python run.py
    ```

3.  **Seed initial data** (optional, run in a separate terminal):
    ```powershell
    # Make sure your virtual environment is active.
    python seed.py
    ```

Login at `http://localhost:5000` with `admin` / `admin123`.

---

### Project Structure

-   **Data structures**: `app/ds/structures.py`
-   **Routes/UI**: `app/main/routes.py`, templates under `app/templates/`
-   **AI Assistant**: `app/ai/assistant.py`
-   **Optional CSV seed**: put `data/medicines.csv` (name,quantity,expiry_date YYYY-MM-DD) then run `python seed.py`.
