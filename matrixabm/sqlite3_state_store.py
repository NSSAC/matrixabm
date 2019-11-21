"""A SQLite3 database file backed state store."""

from .sqlite3_connector import SQLite3Connector
from .state_store import StateStore


class SQLite3Store(StateStore):
    """Sqlite3 state store."""

    def __init__(self, store_name):
        """Initialize."""
        super().__init__(store_name)

        self.update_cache = []

    def handle_update(self, update):
        """Handle incoming update.

        Parameters
        ----------
            update: StateUpdate
                A state update
        """
        self.update_cache.append(update)

    def flush(self):
        """Apply the updates."""
        self.log.info("Sorting %d updates", len(self.update_cache))
        self.update_cache.sort()

        self.log.info("Applying %d updates", len(self.update_cache))
        con = SQLite3Connector.connection()
        with con:
            for update in self.update_cache:
                update.apply(self)

        self.update_cache.clear()

    def execute(self, sql, params=None):
        """Execute the given sql.

        Parameters
        ----------
            sql: sql statement to execute
            params: optional parameters of the sql statement.
        """
        con = SQLite3Connector.connection()
        if params is None:
            con.execute(sql)
        else:
            con.execute(sql, params)

