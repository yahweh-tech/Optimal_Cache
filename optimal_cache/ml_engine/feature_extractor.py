import numpy as np
import pandas as pd
from collections import defaultdict


class FeatureExtractor:
    """
    Extracts temporal, frequency, and statistical features for cache items
    from an access trace up to a specific time step t.
    """

    def __init__(self, window_size=50, alpha=0.3):
        self.window_size = window_size
        self.alpha = alpha  # EMA smoothing factor

    def extract_features_for_item(self, history, item, current_step):
        """
        Extract feature dict for a single item at current_step given access history.
        history: list of items accessed up to current_step - 1.
        item: key/page_id/uri
        current_step: int
        """
        # Find all step indices where item was accessed prior to current_step
        indices = [i for i, val in enumerate(history[:current_step]) if val == item]

        if not indices:
            # Item never seen in history prior to current_step
            return {
                "recency": current_step + 1,
                "freq_window": 0,
                "freq_total": 0,
                "avg_interval": 1000.0,
                "std_interval": 0.0,
                "ema_interval": 1000.0,
                "global_access_ratio": 0.0,
            }

        last_seen = indices[-1]
        recency = current_step - last_seen

        # Frequency in recent window
        recent_window_start = max(0, current_step - self.window_size)
        freq_window = sum(1 for x in history[recent_window_start:current_step] if x == item)
        freq_total = len(indices)

        # Calculate access intervals
        if len(indices) > 1:
            intervals = [indices[i] - indices[i - 1] for i in range(1, len(indices))]
            avg_interval = float(np.mean(intervals))
            std_interval = float(np.std(intervals))

            # Exponential Moving Average of intervals
            ema = float(intervals[0])
            for inv in intervals[1:]:
                ema = self.alpha * inv + (1 - self.alpha) * ema
            ema_interval = float(ema)
        else:
            avg_interval = float(recency)
            std_interval = 0.0
            ema_interval = float(recency)

        global_access_ratio = freq_total / max(1, current_step)

        return {
            "recency": float(recency),
            "freq_window": float(freq_window),
            "freq_total": float(freq_total),
            "avg_interval": float(avg_interval),
            "std_interval": float(std_interval),
            "ema_interval": float(ema_interval),
            "global_access_ratio": float(global_access_ratio),
        }

    def build_dataset_from_trace(self, sequence):
        """
        Generates feature DataFrame X and ground truth y (reuse distance) for a complete sequence.
        sequence: list of items accessed.
        """
        features_list = []
        labels_list = []

        total_steps = len(sequence)
        # Precompute next index map for O(1) or fast lookup of true future reuse distance
        item_positions = defaultdict(list)
        for step, item in enumerate(sequence):
            item_positions[item].append(step)

        for step, item in enumerate(sequence):
            if step == 0:
                continue

            history = sequence[:step]
            # Extract features for current item
            feat = self.extract_features_for_item(history, item, step)

            # Ground truth: next access step of item after 'step'
            all_pos = item_positions[item]
            # Find next position > step
            future_pos = [p for p in all_pos if p > step]
            if future_pos:
                reuse_dist = float(future_pos[0] - step)
            else:
                reuse_dist = float(total_steps - step + 100)  # Large penalty for items not re-accessed

            features_list.append(feat)
            labels_list.append(reuse_dist)

        df_X = pd.DataFrame(features_list)
        df_y = pd.Series(labels_list, name="reuse_distance")
        return df_X, df_y
