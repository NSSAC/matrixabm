"""Shared state store implementations."""

import sqlite3
from abc import ABC, abstractmethod

from sortedcontainers import SortedList

class SharedStateStore(ABC):
    """Interface for shared state stores."""

    def __init__(self, store_name, dsn, init_func=None, maint_func=None):
        """Initialize."""
        self.store_name = store_name
        self.dsn = dsn
        self.init_func = init_func
        self.maint_func = maint_func

    @abstractmethod
    def close(self):
        """Close the store."""

    def run_init(self):
        """Run the initalization function."""
        if self.init_func is not None:
            self.init_func(self)

    def run_maint(self):
        """Run the maintainance function."""
        if self.maint_func is not None:
            self.maint_func(self)

    @abstractmethod
    def handle_update(self, order_key, update):
        """Handle incoming update."""

    @abstractmethod
    def flush(self):
        """Apply the updates to the shared state store."""

# Shared sqlite3 connection
_sqlite3_connection = sqlite3.connect(":memory:")

def get_sqlite3_connection():
    """Return the global sqlite3 connection."""
    return _sqlite3_connection

def _get_first(xs):
    return xs[0]

class SQLite3Store(SharedStateStore):
    """Sqlite3 state store."""

    def __init__(self, *args, **kwargs):
        """Initialize."""
        super().__init__(*args, **kwargs)
        self.con = get_sqlite3_connection()
        self.update_cache = SortedList(key=_get_first)

        sql = f"attach database '{self.dsn}' as {self.store_name}"
        self.con.execute(sql)

    def handle_update(self, order_key, update):
        """Handle incoming update."""
        sql, params = update
        self.update_cache.add((order_key, sql, params))

    def flush(self):
        """Apply the updates to the shared state store."""
        if not self.update_cache:
            return

        with self.con:
            cur = self.con.cursor()
            for _, sql, params in self.update_cache:
                if params is None:
                    cur.execute(sql)
                else:
                    cur.execute(sql, tuple(params))

        self.update_cache = SortedList(key=_get_first)

    def close(self):
        """Close the store."""
        self.flush()
