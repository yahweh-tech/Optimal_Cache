import json
import pandas as pd
from django.shortcuts import render
from django.http import JsonResponse
from django.views.decorators.csrf import csrf_exempt

from ml_engine.dataset_generator import DatasetGenerator
from ml_engine.feature_extractor import FeatureExtractor
from ml_engine.xgboost_model import XGBoostCacheModel
from cache_engine.simulator import CacheSimulatorRunner


def dashboard_view(request):
    """Renders the main interactive web dashboard."""
    return render(request, "index.html")


@csrf_exempt
def api_simulate(request):
    """
    REST API endpoint to run cache simulations comparing LRU, LFU,
    Belady's Optimal, and XGBoost Predictive Optimal algorithms.
    """
    if request.method != "POST":
        return JsonResponse({"error": "Only POST requests are supported."}, status=405)

    try:
        body = json.loads(request.body.decode("utf-8")) if request.body else {}
        pattern = body.get("pattern", "zipf")
        cache_size = int(body.get("cache_size", 10))
        trace_length = int(body.get("trace_length", 500))
        custom_sequence = body.get("custom_sequence", None)

        if custom_sequence:
            if isinstance(custom_sequence, str):
                sequence = [x.strip() for x in custom_sequence.split(",") if x.strip()]
            else:
                sequence = [str(x) for x in custom_sequence]
        else:
            if pattern == "zipf":
                sequence = DatasetGenerator.generate_zipf_trace(length=trace_length, num_items=50)
            elif pattern == "temporal":
                sequence = DatasetGenerator.generate_temporal_locality_trace(length=trace_length, num_items=50)
            elif pattern == "api_endpoints":
                sequence = DatasetGenerator.generate_api_endpoint_trace(length=trace_length)
            else:
                sequence = DatasetGenerator.generate_zipf_trace(length=trace_length, num_items=50)

        runner = CacheSimulatorRunner(cache_size=cache_size)
        results = runner.run_benchmark(sequence)

        return JsonResponse({"success": True, "results": results})

    except Exception as e:
        return JsonResponse({"success": False, "error": str(e)}, status=500)


# =====================================================================
# LIVE REAL-TIME CACHING ENGINE INSTANCE
# =====================================================================
class LiveXGBoostCacheManager:
    def __init__(self, cache_size=5):
        self.cache_size = cache_size
        self.cache = []
        self.history = []
        self.feature_extractor = FeatureExtractor()
        self.xgb_model = XGBoostCacheModel(n_estimators=50, max_depth=4)
        self.total_requests = 0
        self.hits = 0
        self.misses = 0

        # Initial bootstrap training on a default sample trace
        boot_trace = DatasetGenerator.generate_api_endpoint_trace(length=200)
        X_df, y_df = self.feature_extractor.build_dataset_from_trace(boot_trace)
        self.xgb_model.train(X_df, y_df)

    def access_key(self, key):
        self.total_requests += 1
        current_step = len(self.history)
        self.history.append(key)

        # Cache Hit
        if key in self.cache:
            self.hits += 1
            return {
                "status": "HIT",
                "key": key,
                "evicted": None,
                "current_cache": self.cache.copy(),
                "hit_ratio": round(self.hits / self.total_requests, 4),
            }

        # Cache Miss
        self.misses += 1
        evicted = None
        predictions = {}

        if len(self.cache) < self.cache_size:
            self.cache.append(key)
        else:
            # Predict reuse distance for all keys currently in live cache
            feat_rows = []
            for cached_item in self.cache:
                feat = self.feature_extractor.extract_features_for_item(self.history, cached_item, current_step)
                feat_rows.append(feat)

            df_feat = pd.DataFrame(feat_rows)
            preds = self.xgb_model.predict(df_feat)

            for cached_item, p_dist in zip(self.cache, preds):
                predictions[str(cached_item)] = round(float(p_dist), 2)

            victim_idx = int(preds.argmax())
            evicted = self.cache[victim_idx]

            self.cache.remove(evicted)
            self.cache.append(key)

        return {
            "status": "MISS",
            "key": key,
            "evicted": evicted,
            "predicted_reuse_distances": predictions,
            "current_cache": self.cache.copy(),
            "hit_ratio": round(self.hits / self.total_requests, 4),
        }


# Singleton live cache manager
LIVE_CACHE_SYSTEM = LiveXGBoostCacheManager(cache_size=5)


@csrf_exempt
def api_cache_get(request):
    """
    Live real-time caching API endpoint (`/api/cache/get/?key=...`).
    Maintains an active in-memory XGBoost predictive cache.
    """
    key = request.GET.get("key", None)
    if not key:
        return JsonResponse({"error": "Query parameter 'key' is required."}, status=400)

    result = LIVE_CACHE_SYSTEM.access_key(key)
    return JsonResponse(result)
