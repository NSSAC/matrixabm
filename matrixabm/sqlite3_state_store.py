"""A SQLite3 database file backed state store."""

from . import INFO_FINE
from .sqlite3_connector import SQLite3Connector
from .state_store import StateStore


class SQLite3Store(StateStore):
    """Sqlite3 state store."""

    def __init__(self, store_name):
        """Initialize."""
        super().__init__(store_name)


        self._insert_sql_cache = {}
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
        self.log.log(INFO_FINE, "Sorting %d updates", len(self.update_cache))
        self.update_cache.sort()

        self.log.log(INFO_FINE, "Applying %d updates", len(self.update_cache))
        con = SQLite3Connector.connection()
        with con:
            for update in self.update_cache:
                update.apply(self)

        self.update_cache.clear()

    def execute(self, sql, params=None):
        """Execute the given sql.

        Parameters
        ----------
        sql: str
            SQL statement to execute
        params: tuple, optional
            Optional parameters of the sql statement.
        """
        con = SQLite3Connector.connection()
        try:
            if params is None:
                return con.execute(sql)
            else:
                return con.execute(sql, params)
        except Exception:
            self.log.error("Error executing sql:\n%s\nparams=%r", sql, params)
            raise

    def insert(self, table, *params):
        """Execute an insert statement.

        Parameters
        ----------
        table: str
            Table to insert into.
        values: tuple
            Values to insert into table.
        """
        if table in self._insert_sql_cache:
            sql = self._insert_sql_cache[table]
        else:
            sql = "insert into %s.%s values (%s)"
            marks = ["?"] * len(params)
            marks = ",".join(marks)
            sql = sql % (self.store_name, table, marks)
            self._insert_sql_cache[table] = sql

        return self.execute(sql, params)
