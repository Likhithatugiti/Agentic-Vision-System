"""
EventStore — thread-safe in-memory circular buffer for triggered alerts.

Provides:
  add(alert)        — append a new alert
  recent(n)         — return last n alerts, newest first
  clear()           — reset the store
  stats()           — aggregate counts by severity
"""

import threading
from collections import deque, Counter
from typing import List


class EventStore:
    def __init__(self, maxlen: int = 500):
        self._store: deque = deque(maxlen=maxlen)
        self._lock = threading.Lock()

    def add(self, alert: dict):
        with self._lock:
            self._store.append(alert)

    def recent(self, n: int = 50) -> List[dict]:
        with self._lock:
            items = list(self._store)
        return list(reversed(items))[:n]

    def clear(self):
        with self._lock:
            self._store.clear()

    def stats(self) -> dict:
        with self._lock:
            items = list(self._store)
        severity_counts = Counter(a.get("severity") for a in items)
        rule_counts = Counter(a.get("primary_rule") for a in items)
        return {
            "total_alerts": len(items),
            "by_severity":  dict(severity_counts),
            "by_rule":      dict(rule_counts),
        }
