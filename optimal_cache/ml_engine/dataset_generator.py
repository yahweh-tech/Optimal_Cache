import random
import numpy as np


class DatasetGenerator:
    """
    Generates synthetic and realistic data access sequences/traces.
    Supports Zipfian distribution, temporal locality, looping patterns, and REST API URIs.
    """

    @staticmethod
    def generate_zipf_trace(length=1000, num_items=50, alpha=1.2, seed=42):
        """
        Zipfian distribution: A small subset of items (e.g. 20%) gets a large portion (80%) of accesses.
        """
        np.random.seed(seed)
        ranks = np.arange(1, num_items + 1)
        weights = 1.0 / (ranks ** alpha)
        probabilities = weights / np.sum(weights)

        # Generate page IDs as numbers or strings
        items = [f"Item_{i}" for i in range(1, num_items + 1)]
        sequence = list(np.random.choice(items, size=length, p=probabilities))
        return sequence

    @staticmethod
    def generate_temporal_locality_trace(length=1000, num_items=50, locality_prob=0.7, seed=42):
        """
        Temporal locality: Recently accessed items are much more likely to be accessed again soon.
        """
        random.seed(seed)
        items = [f"Page_{i}" for i in range(1, num_items + 1)]
        sequence = []
        recent_window = []

        for _ in range(length):
            if recent_window and random.random() < locality_prob:
                item = random.choice(recent_window)
            else:
                item = random.choice(items)

            sequence.append(item)
            recent_window.append(item)
            if len(recent_window) > 10:
                recent_window.pop(0)

        return sequence

    @staticmethod
    def generate_api_endpoint_trace(length=1000, seed=42):
        """
        Simulates realistic REST API call traffic for web application caching.
        """
        random.seed(seed)
        endpoints = [
            "/api/v1/products/list",
            "/api/v1/users/profile",
            "/api/v1/cart/checkout",
            "/api/v1/analytics/dashboard",
            "/api/v1/products/42",
            "/api/v1/products/108",
            "/api/v1/categories/electronics",
            "/api/v1/orders/history",
            "/api/v1/search?q=phone",
            "/api/v1/recommendations",
        ]
        weights = [0.30, 0.20, 0.05, 0.02, 0.15, 0.10, 0.08, 0.04, 0.04, 0.02]
        weights = np.array(weights) / np.sum(weights)

        sequence = list(np.random.choice(endpoints, size=length, p=weights))
        return sequence
