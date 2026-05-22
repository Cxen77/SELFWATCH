import time
import numpy as np

class IdentityMemory:
    """Long-term ReID gallery per identity for re-identification after track loss."""

    def __init__(self, match_threshold=0.80, expire_seconds=300,
                 gallery_size=15, confirm_hits=3):
        self.entries = {}
        self.match_threshold = match_threshold
        self.expire_seconds = expire_seconds
        self.gallery_size = gallery_size
        self.confirm_hits = confirm_hits
        self._confirm_buf = {}

    def query(self, embedding, candidate_key=None):
        if embedding is None:
            return None, 0.0
        self._expire()
        best_gid, best_sim = None, 0.0
        for gid, entry in self.entries.items():
            sim = max(float(np.dot(embedding, g) / (np.linalg.norm(embedding) * np.linalg.norm(g) + 1e-6))
                      for g in entry['gallery'])
            if sim > best_sim:
                best_sim, best_gid = sim, gid
        if best_sim < self.match_threshold:
            self._confirm_buf.pop(candidate_key, None)
            return None, 0.0
        if candidate_key is not None:
            buf = self._confirm_buf.get(candidate_key, {'gid': None, 'count': 0})
            if buf['gid'] == best_gid:
                buf['count'] += 1
            else:
                buf = {'gid': best_gid, 'count': 1}
            self._confirm_buf[candidate_key] = buf
            if buf['count'] < self.confirm_hits:
                return None, 0.0
            del self._confirm_buf[candidate_key]
        return best_gid, best_sim

    def store(self, gid, embedding, label="object"):
        if embedding is None:
            return
        if gid in self.entries:
            g = self.entries[gid]['gallery']
            g.append(embedding.copy())
            if len(g) > self.gallery_size:
                g.pop(0)
            self.entries[gid]['last_seen'] = time.time()
            if label != "object":
                self.entries[gid]['label'] = label
        else:
            self.entries[gid] = {'gallery': [embedding.copy()],
                                  'label': label, 'last_seen': time.time()}

    def get_label(self, gid):
        return self.entries.get(gid, {}).get('label', 'object')

    def _expire(self):
        now = time.time()
        for gid in [g for g, e in self.entries.items()
                    if now - e['last_seen'] > self.expire_seconds]:
            del self.entries[gid]
