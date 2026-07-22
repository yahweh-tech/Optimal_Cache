class OptimalCache:

    def __init__(self, cache_size):
        self.cache_size = cache_size

    def simulate(self, sequence):

        cache = []

        hits = 0
        misses = 0
        history = []

        for i, page in enumerate(sequence):

            # Hit
            if page in cache:

                hits += 1

                history.append({
                    "page": page,
                    "cache": cache.copy(),
                    "action": "HIT"
                })

                continue

            misses += 1

            # Space available
            if len(cache) < self.cache_size:

                cache.append(page)

            else:

                future = sequence[i + 1:]

                farthest = -1
                victim = None

                for cached_page in cache:

                    if cached_page not in future:

                        victim = cached_page
                        break

                    next_use = future.index(cached_page)

                    if next_use > farthest:

                        farthest = next_use
                        victim = cached_page

                cache.remove(victim)
                cache.append(page)

            history.append({
                "page": page,
                "cache": cache.copy(),
                "action": "MISS"
            })

        return {

            "hits": hits,

            "misses": misses,

            "hit_ratio": hits / len(sequence),

            "history": history
        }