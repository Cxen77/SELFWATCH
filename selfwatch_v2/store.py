"""Append-only observations with retry idempotency and conflict detection."""
import json
import sqlite3
from pathlib import Path

from .contracts import Observation


class ObservationStore:
    def __init__(self, path):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        self.connection = sqlite3.connect(path)
        self.connection.execute("PRAGMA journal_mode=WAL")
        self.connection.execute("""CREATE TABLE IF NOT EXISTS observations (
            observation_id TEXT PRIMARY KEY, captured_at REAL NOT NULL,
            camera_id TEXT NOT NULL, payload TEXT NOT NULL)""")
        self.connection.execute("CREATE INDEX IF NOT EXISTS obs_time ON observations(captured_at, camera_id)")
        self.connection.commit()

    def append(self, observation: Observation):
        payload = json.dumps(observation.to_dict(), sort_keys=True, allow_nan=False)
        key = observation.frame.observation_id
        with self.connection:
            existing = self.connection.execute(
                "SELECT payload FROM observations WHERE observation_id=?", (key,)
            ).fetchone()
            if existing:
                if existing[0] != payload:
                    raise ValueError(f"Conflicting observation for {key}")
                return False
            self.connection.execute("INSERT INTO observations VALUES (?, ?, ?, ?)",
                (key, observation.frame.captured_at_s, observation.frame.camera_id, payload))
        return True

    def records(self):
        for (payload,) in self.connection.execute(
                "SELECT payload FROM observations ORDER BY captured_at, camera_id, observation_id"):
            yield json.loads(payload)

    def close(self):
        self.connection.close()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
