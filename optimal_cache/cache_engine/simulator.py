from cache_engine.lru import LRUCache
from cache_engine.lfu import LFUCache
from cache_engine.optimal import OptimalCache
from ml_engine.feature_extractor import FeatureExtractor
from ml_engine.dataset_generator import DatasetGenerator
from ml_engine.xgboost_model import XGBoostCacheModel
from ml_engine.predictive_optimal import MLPredictiveOptimalCache


class CacheSimulatorRunner:
    """
    Benchmark runner comparing LRU, LFU, Belady's Theoretical Optimal,
    and XGBoost Predictive Optimal algorithms.
    """

    def __init__(self, cache_size=10):
        self.cache_size = cache_size

    def run_benchmark(self, sequence, train_xgb=True):
        total_requests = len(sequence)

        # 1. LRU Simulation
        lru = LRUCache(self.cache_size)
        res_lru = lru.simulate(sequence)

        # 2. LFU Simulation
        lfu = LFUCache(self.cache_size)
        res_lfu = lfu.simulate(sequence)

        # 3. Belady's Optimal (Oracle) Simulation
        opt = OptimalCache(self.cache_size)
        res_opt = opt.simulate(sequence)

        # 4. XGBoost Predictive Optimal
        fe = FeatureExtractor()
        df_X, df_y = fe.build_dataset_from_trace(sequence)

        xgb_model = XGBoostCacheModel(n_estimators=80, max_depth=5, learning_rate=0.1)
        train_metrics = xgb_model.train(df_X, df_y)

        ml_cache = MLPredictiveOptimalCache(self.cache_size, xgb_model, fe)
        res_ml = ml_cache.simulate(sequence)

        # Calculate Efficiency & Proximity Ratios
        belady_hit_ratio = res_opt["hit_ratio"]
        ml_hit_ratio = res_ml["hit_ratio"]
        lru_hit_ratio = res_lru["hit_ratio"]

        proximity_to_belady = (
            round((ml_hit_ratio / belady_hit_ratio) * 100, 2)
            if belady_hit_ratio > 0
            else 100.0
        )

        improvement_over_lru = (
            round(((ml_hit_ratio - lru_hit_ratio) / lru_hit_ratio) * 100, 2)
            if lru_hit_ratio > 0
            else 0.0
        )

        return {
            "parameters": {
                "cache_size": self.cache_size,
                "total_requests": total_requests,
                "unique_items": len(set(sequence)),
            },
            "comparison": {
                "LRU": {
                    "hits": res_lru["hits"],
                    "misses": res_lru["misses"],
                    "hit_ratio": round(res_lru["hit_ratio"], 4),
                },
                "LFU": {
                    "hits": res_lfu["hits"],
                    "misses": res_lfu["misses"],
                    "hit_ratio": round(res_lfu["hit_ratio"], 4),
                },
                "Belady_Optimal": {
                    "hits": res_opt["hits"],
                    "misses": res_opt["misses"],
                    "hit_ratio": round(res_opt["hit_ratio"], 4),
                },
                "XGBoost_Predictive": {
                    "hits": res_ml["hits"],
                    "misses": res_ml["misses"],
                    "hit_ratio": round(res_ml["hit_ratio"], 4),
                },
            },
            "research_findings": {
                "belady_proximity_percentage": proximity_to_belady,
                "improvement_over_lru_percentage": improvement_over_lru,
                "model_metrics": train_metrics,
                "feature_importance": xgb_model.feature_importances_,
            },
            "detailed_histories": {
                "lru": res_lru["history"][:50],  # preview first 50 steps
                "xgboost": res_ml["history"][:50],
                "belady": res_opt["history"][:50],
            },
        }
