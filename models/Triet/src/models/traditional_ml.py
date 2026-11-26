"""Scikit-learn traditional models for ASL comparison experiments."""

from __future__ import annotations

from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.calibration import CalibratedClassifierCV


def create_knn(n_neighbors: int = 5) -> Pipeline:
    return Pipeline([
        ("standard_scaler", StandardScaler()),
        ("knn", KNeighborsClassifier(n_neighbors=n_neighbors)),
    ])


def create_svm() -> Pipeline:
    return Pipeline([
        ("standard_scaler", StandardScaler()),
        ("svm", CalibratedClassifierCV(
            estimator=SVC(kernel="rbf"),
            cv=3,
        )),
    ])


def create_random_forest(n_estimators: int = 200) -> Pipeline:
    return Pipeline([
        ("standard_scaler", StandardScaler()),
        ("random_forest", RandomForestClassifier(n_estimators=n_estimators, random_state=42)),
    ])
