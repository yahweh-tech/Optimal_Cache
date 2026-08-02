import os
import joblib
import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from xgboost import XGBRegressor


class XGBoostCacheModel:
    """
    XGBoost Machine Learning model to predict future reuse distances
    for cache eviction candidates.
    """

    def __init__(self, n_estimators=100, max_depth=6, learning_rate=0.1, random_state=42):
        self.model = XGBRegressor(
            n_estimators=n_estimators,
            max_depth=max_depth,
            learning_rate=learning_rate,
            random_state=random_state,
            objective="reg:squarederror",
            n_jobs=-1,
        )
        self.is_trained = False
        self.feature_names = [
            "recency",
            "freq_window",
            "freq_total",
            "avg_interval",
            "std_interval",
            "ema_interval",
            "global_access_ratio",
        ]
        self.metrics = {}
        self.feature_importances_ = {}

    def train(self, X, y):
        """
        Trains the XGBoost Regressor on feature DataFrame X and target Series y.
        Calculates MAE, RMSE, R2, and feature importance scores.
        """
        if isinstance(X, dict):
            X = pd.DataFrame([X])

        self.model.fit(X, y)
        self.is_trained = True

        # Calculate metrics on training dataset
        preds = self.model.predict(X)
        rmse = float(np.sqrt(mean_squared_error(y, preds)))
        mae = float(mean_absolute_error(y, preds))
        r2 = float(r2_score(y, preds))

        self.metrics = {
            "rmse": round(rmse, 4),
            "mae": round(mae, 4),
            "r2_score": round(r2, 4),
        }

        # Calculate feature importances
        importances = self.model.feature_importances_
        self.feature_importances_ = {
            name: round(float(imp), 4)
            for name, imp in zip(self.feature_names, importances)
        }

        return self.metrics

    def predict(self, feature_df_or_dict):
        """
        Predicts future reuse distance for one or multiple items.
        """
        if not self.is_trained:
            raise ValueError("XGBoostCacheModel must be trained before predicting.")

        if isinstance(feature_df_or_dict, dict):
            df = pd.DataFrame([feature_df_or_dict])[self.feature_names]
        elif isinstance(feature_df_or_dict, pd.DataFrame):
            df = feature_df_or_dict[self.feature_names]
        else:
            df = pd.DataFrame(feature_df_or_dict, columns=self.feature_names)

        preds = self.model.predict(df)
        return preds

    def save(self, filepath):
        """Saves model to disk."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        joblib.dump(
            {
                "model": self.model,
                "is_trained": self.is_trained,
                "metrics": self.metrics,
                "feature_importances_": self.feature_importances_,
            },
            filepath,
        )

    def load(self, filepath):
        """Loads model from disk."""
        data = joblib.load(filepath)
        self.model = data["model"]
        self.is_trained = data["is_trained"]
        self.metrics = data["metrics"]
        self.feature_importances_ = data["feature_importances_"]
