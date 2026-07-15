class Tokenizer:
    def __init__(self, vocab_size=1024):
        self.vocab_size = vocab_size
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        
    def train(self, text):
        # 300,000 bytes is more than enough to capture Devanagari and English pairs rapidly
        train_bytes = text[:300000].encode('utf-8')
        ids = list(train_bytes)
        num_merges = self.vocab_size - 256
        
        for i in range(num_merges):
            stats = self._get_stats(ids)
            if not stats:
                break
            best_pair = max(stats, key=stats.get)
            new_id = 256 + i
            
            self.merges[best_pair] = new_id
            self.vocab[new_id] = self.vocab[best_pair[0]] + self.vocab[best_pair[1]]
            ids = self._merge(ids, best_pair, new_id)
        
    def _get_stats(self, ids):
        counts = {}
        for pair in zip(ids, ids[1:]):
            counts[pair] = counts.get(pair, 0) + 1
        return counts

    def _merge(self, ids, pair, idx):
        newids = []
        i = 0
        while i < len(ids):
            if i < len(ids) - 1 and ids[i] == pair[0] and ids[i+1] == pair[1]:
                newids.append(idx)
                i += 2
            else:
                newids.append(ids[i])
                i += 1
        return newids

    def encode(self, text):
        # OPTIMIZATION: Chunk the text to prevent massive slowdowns on the 7MB corpus
        b = text.encode("utf-8")
        chunk_size = 50000 # Process in 50KB chunks
        final_ids = []
        
        for c in range(0, len(b), chunk_size):
            chunk_bytes = b[c:c+chunk_size]
            ids = list(chunk_bytes)
            while len(ids) >= 2:
                stats = self._get_stats(ids)
                if not stats:
                    break
                # Find pair with lowest merge index (first merged)
                pair = min(stats.keys(), key=lambda p: self.merges.get(p, float("inf")))
                if pair not in self.merges:
                    break 
                idx = self.merges[pair]
                ids = self._merge(ids, pair, idx)
            final_ids.extend(ids)
            
        return final_ids

    def decode(self, ids):
        b = b"".join(self.vocab[idx] for idx in ids)
        return b.decode("utf-8", errors="replace")

    def load(self, merges_dict=None):
        """Satisfies the required interface rule to encode arbitrary UTF-8 text."""
        if merges_dict is not None:
            self.merges = merges_dict
            # Rebuild vocab from loaded merges
            for (p0, p1), idx in self.merges.items():
                self.vocab[idx] = self.vocab[p0] + self.vocab[p1]