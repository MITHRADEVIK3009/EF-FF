
import sys, json, math
from collections import deque
import numpy as np

# sklearn is used only for the forest (no partial_fit)
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

class HybridWorker:
    def __init__(self):
        self.feature_cols = []
        self.target_col = None          # optional; if None and "Skip_Rate" exists, we derive target from median_skip
        self.retain_size = 500
        self.n_trees = 20
        self.corr_threshold = 0.3
        self.warmup_size = 200

        self.buffer_X = deque(maxlen=self.retain_size)
        self.buffer_y = deque(maxlen=self.retain_size)

        self.forest = None
        self.median_skip = None         # used to derive target if target_col is None
        self.last_metrics = None
        self.seen = 0

    def init(self, cfg):
        self.feature_cols = cfg.get("feature_cols", [])
        self.target_col = cfg.get("target_col")   # may be None
        self.retain_size = int(cfg.get("retain_size", 500))
        self.n_trees = int(cfg.get("n_trees", 20))
        self.corr_threshold = float(cfg.get("correlation_threshold", 0.3))
        self.warmup_size = int(cfg.get("warmup_size", 200))

        self.buffer_X = deque(maxlen=self.retain_size)
        self.buffer_y = deque(maxlen=self.retain_size)
        self.forest = None
        self.median_skip = None
        self.last_metrics = None
        self.seen = 0

        return {"ok": True, "msg": "initialized"}

    def _rows_to_Xy(self, rows):
        # Convert list[dict] -> (X, y). If target_col missing, derive from Skip_Rate median once warmup done.
        X_list, y_list, skip_vals = [], [], []

        for r in rows:
            try:
                x = [float(r[c]) for c in self.feature_cols]
            except KeyError as e:
                raise ValueError(f"Missing feature column in row: {e}")
            X_list.append(x)

            if self.target_col and self.target_col in r:
                y_list.append(int(r[self.target_col]))
            else:
                # derive from skip rate if available
                sr = None
                if "Skip_Rate" in r:
                    sr = float(r["Skip_Rate"])
                skip_vals.append(sr)
                # label will be assigned after we have/update median
                y_list.append(sr)

        X = np.asarray(X_list, dtype=float)
        y = np.asarray(y_list, dtype=float)  # may contain skip rates temporarily
        return X, y, np.asarray([v for v in skip_vals if v is not None], dtype=float)

    def _ensure_labels(self, y_array):
        """If target_col is None, convert skip rates into binary labels using median_skip."""
        if self.target_col:
            return y_array.astype(int)

        if self.median_skip is None:
            # compute median from whatever we have seen so far in buffer + current
            # collect skip values from buffer_y (when target_col None, buffer_y stores binary already)
            raise RuntimeError("median_skip not set; warmup required with Skip_Rate to compute threshold.")

        # y_array currently contains skip rates; convert
        return (y_array > self.median_skip).astype(int)

    def _update_median_skip(self, rows):
        # collect Skip_Rate values
        vals = [float(r["Skip_Rate"]) for r in rows if "Skip_Rate" in r]
        if not vals:
            return
        if self.median_skip is None:
            self.median_skip = float(np.median(vals))
        else:
            # incremental median approximation: combine a small window with current median
            # (kept simple; robust enough for streaming)
            mixed = np.array(vals + [self.median_skip]*max(1, len(vals)//5))
            self.median_skip = float(np.median(mixed))

    def _train_forest(self):
        if len(self.buffer_X) < max(50, min(200, self.retain_size//5)):
            return False  # not enough data
        X = np.asarray(self.buffer_X, dtype=float)
        y = np.asarray(self.buffer_y, dtype=int)
        max_depth = max(2, int(math.log2(max(2, len(X)))))
        self.forest = RandomForestClassifier(
            n_estimators=self.n_trees, max_depth=max_depth, random_state=42
        )
        self.forest.fit(X, y)
        return True

    def _edgeframe(self, X):
        if X.shape[0] < 2:
            return {"nodes":0, "edges":0, "sparsity_ratio":1.0}
        corr = np.corrcoef(X.T)
        p = corr.shape[0]
        edges = 0
        for i in range(p):
            for j in range(i+1, p):
                if abs(corr[i, j]) > self.corr_threshold:
                    edges += 1
        total_possible = p*(p-1)//2
        sparsity = 1 - (edges/total_possible if total_possible else 1.0)
        return {"nodes": p, "edges": edges, "sparsity_ratio": float(sparsity)}

    def warmup(self, rows):
        if self.target_col is None and all(("Skip_Rate" in r) for r in rows):
            self._update_median_skip(rows)

        X, y_raw, _ = self._rows_to_Xy(rows)
        if self.target_col is None:
            if self.median_skip is None:
                raise RuntimeError("Warmup requires Skip_Rate to compute median label threshold.")
            y = self._ensure_labels(y_raw)
        else:
            y = y_raw.astype(int)

        for i in range(len(X)):
            self.buffer_X.append(X[i])
            self.buffer_y.append(y[i])

        self._train_forest()
        return {"ok": True, "buffer": len(self.buffer_X)}

    def update(self, rows):
        self.seen += len(rows)
        if self.target_col is None and any(("Skip_Rate" in r) for r in rows):
            self._update_median_skip(rows)

        X_batch, y_raw, _ = self._rows_to_Xy(rows)
        y_batch = self._ensure_labels(y_raw)

        # Evaluate on current model before training (prequential evaluation)
        if self.forest is not None and len(X_batch) > 0:
            y_pred = self.forest.predict(X_batch)
            acc = float(accuracy_score(y_batch, y_pred)) * 100.0
        else:
            acc = float("nan")

        # Update buffer with batch
        for i in range(len(X_batch)):
            self.buffer_X.append(X_batch[i])
            self.buffer_y.append(y_batch[i])

        # Retrain forest on retained buffer
        trained = self._train_forest()

        # EdgeFrame on retained buffer
        X_ret = np.asarray(self.buffer_X, dtype=float)
        ef = self._edgeframe(X_ret)

        metrics = {
            "seen": self.seen,
            "accuracy": acc,
            "retained_size": len(self.buffer_X),
            "memory_usage_mb": (len(self.buffer_X) * X_ret.shape[1] * 8) / (1024*1024) if X_ret.size else 0.0,
            "forest_size": self.n_trees if trained else 0,
            "edgeframe_nodes": ef["nodes"],
            "edgeframe_edges": ef["edges"],
            "sparsity_ratio": ef["sparsity_ratio"],
            "median_skip": self.median_skip,
        }
        self.last_metrics = metrics
        return {"ok": True, "metrics": metrics}

    def status(self):
        return {"ok": True, "metrics": self.last_metrics}

    def reset(self):
        return self.init({
            "feature_cols": self.feature_cols,
            "target_col": self.target_col,
            "retain_size": self.retain_size,
            "n_trees": self.n_trees,
            "correlation_threshold": self.corr_threshold,
            "warmup_size": self.warmup_size
        })


# Main execution loop (only runs when script is executed directly)
if __name__ == "__main__":
    worker = HybridWorker()

    def write(obj):
        sys.stdout.write(json.dumps(obj) + "\n")
        sys.stdout.flush()

    for line in sys.stdin:
        if not line.strip():
            continue
        try:
            msg = json.loads(line)
            cmd = msg.get("cmd")
            if cmd == "init":
                out = worker.init(msg.get("config", {}))
            elif cmd == "warmup":
                out = worker.warmup(msg.get("data", []))
            elif cmd == "update":
                out = worker.update(msg.get("data", []))
            elif cmd == "status":
                out = worker.status()
            elif cmd == "reset":
                out = worker.reset()
            else:
                out = {"ok": False, "error": f"unknown cmd {cmd}"}
        except Exception as e:
            out = {"ok": False, "error": str(e)}
        write(out)
