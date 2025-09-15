"""Minimal stub for ML model manager used by the web routes.

This file provides a small, dependency-free fallback implementation so the
application can start even when the original heavy ML modules were removed.

It intentionally returns safe defaults so endpoints depending on ML features
don't crash; later you can replace this with your real ML implementation.
"""
from typing import Any, Dict, List


class _DummyModel:
    def __init__(self, user_id: int = None):
        self.user_id = user_id

    def predict_demand(self, medicine_id: int, days: int = 7) -> List[float]:
        # Return zeros for requested days (safe default)
        return [0.0 for _ in range(days)]

    def detect_anomalies(self, days: int = 7) -> List[Dict[str, Any]]:
        return []

    def predict_clusters(self) -> List[Dict[str, Any]]:
        return []


class MLModelManager:
    """Lightweight stub of the ML management API expected by routes.

    Methods mirror the real implementation surface used by the Flask routes so
    the app can import this module safely in production if the full ML code
    isn't available.
    """

    def __init__(self, user_id: int):
        self.user_id = user_id
        self.demand_model = _DummyModel(user_id)
        self.anomaly_model = _DummyModel(user_id)
        self.clustering_model = _DummyModel(user_id)

    def train_all_models(self) -> Dict[str, Any]:
        # No-op training placeholder
        return {"status": "skipped", "reason": "ML not installed in this deployment"}

    def get_ml_insights(self) -> Dict[str, Any]:
        # Return an empty insights structure
        return {"models": {}, "insights": {}}
