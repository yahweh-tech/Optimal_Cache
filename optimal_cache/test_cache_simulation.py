import sys
import os

# Add optimal_cache directory to path
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from ml_engine.dataset_generator import DatasetGenerator
from cache_engine.simulator import CacheSimulatorRunner


def main():
    print("=" * 60)
    print("  OPTIMAL CACHE ENGINE: XGBOOST PREDICTIVE BENCHMARK  ")
    print("=" * 60)

    # Generate sample Zipf sequence
    sequence = DatasetGenerator.generate_zipf_trace(length=300, num_items=30, alpha=1.2, seed=42)
    print(f"Generated trace with {len(sequence)} requests across {len(set(sequence))} unique items.\n")

    runner = CacheSimulatorRunner(cache_size=8)
    results = runner.run_benchmark(sequence)

    comp = results["comparison"]
    findings = results["research_findings"]

    print("CACHE HIT RATIO COMPARISON:")
    print(f"  - LRU Cache Hit Ratio              : {comp['LRU']['hit_ratio'] * 100:.2f}%")
    print(f"  - LFU Cache Hit Ratio              : {comp['LFU']['hit_ratio'] * 100:.2f}%")
    print(f"  - XGBoost Predictive Optimal Ratio : {comp['XGBoost_Predictive']['hit_ratio'] * 100:.2f}%")
    print(f"  - Belady's Theoretical Optimal     : {comp['Belady_Optimal']['hit_ratio'] * 100:.2f}%\n")

    print("RESEARCH FINDINGS & ACCURACY:")
    print(f"  - Belady Proximity Ratio          : {findings['belady_proximity_percentage']}%")
    print(f"  - Hit Ratio Improvement over LRU   : {findings['improvement_over_lru_percentage']}%\n")

    print("XGBOOST MODEL PERFORMANCE METRICS:")
    for k, v in findings["model_metrics"].items():
        print(f"  - {k.upper():12s}: {v}")

    print("\nFEATURE IMPORTANCE BREAKDOWN:")
    for feat, imp in findings["feature_importance"].items():
        print(f"  - {feat:20s}: {imp:.4f}")

    print("=" * 60)
    print("SUCCESS: Benchmark executed cleanly.")


if __name__ == "__main__":
    main()
