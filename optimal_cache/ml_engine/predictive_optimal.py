import pandas as pd
from ml_engine.feature_extractor import FeatureExtractor


class MLPredictiveOptimalCache:
    """
    XGBoost-powered Predictive Optimal Cache Eviction Engine.
    Employs Belady's Optimal Strategy using ML predictions when future is unknown.
    """

    def __init__(self, cache_size, xgb_model, feature_extractor=None):
        self.cache_size = cache_size
        self.xgb_model = xgb_model
        self.feature_extractor = feature_extractor or FeatureExtractor()

    def simulate(self, sequence):
        cache = []
        hits = 0
        misses = 0
        history_log = []
        evictions = []

        history = []

        for step, page in enumerate(sequence):
            history.append(page)

            # Cache Hit
            if page in cache:
                hits += 1
                history_log.append({
                    "step": step,
                    "page": page,
                    "cache": cache.copy(),
                    "action": "HIT",
                    "evicted": None,
                    "predictions": {},
                })
                continue

            # Cache Miss
            misses += 1
            evicted = None
            predictions = {}

            # Cache has free space
            if len(cache) < self.cache_size:
                cache.append(page)
            else:
                # Eviction needed: Extract features for all items currently in cache
                feat_rows = []
                for cached_item in cache:
                    feat = self.feature_extractor.extract_features_for_item(history, cached_item, step)
                    feat_rows.append(feat)

                df_feat = pd.DataFrame(feat_rows)
                predicted_distances = self.xgb_model.predict(df_feat)

                for item, p_dist in zip(cache, predicted_distances):
                    predictions[str(item)] = round(float(p_dist), 2)

                # Select victim item with the maximum predicted reuse distance (farthest in future)
                victim_idx = int(predicted_distances.argmax())
                evicted = cache[victim_idx]

                cache.remove(evicted)
                cache.append(page)
                evictions.append({
                    "step": step,
                    "evicted": evicted,
                    "inserted": page,
                    "predictions": predictions,
                })

            history_log.append({
                "step": step,
                "page": page,
                "cache": cache.copy(),
                "action": "MISS",
                "evicted": evicted,
                "predictions": predictions,
            })

        hit_ratio = hits / max(1, len(sequence))

        return {
            "hits": hits,
            "misses": misses,
            "hit_ratio": round(hit_ratio, 4),
            "evictions_count": len(evictions),
            "history": history_log,
        }
