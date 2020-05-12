"""A SQLite3 database file backed state store."""

from . import asys, INFO_FINE
from .state_store import StateStore


class SQLite3Store(StateStore):
    """Sqlite3 state store."""

    def __init__(self, store_name, connector_name):
        """Initialize."""
        super().__init__(store_name)

        self.connector_name = connector_name
        self._insert_sql_cache = {}
        self._insert_or_ignore_sql_cache = {}
        self.update_cache = []

    def connection(self):
        """Get the connection to the local sqlite3 connector object."""
        return asys.local_actor(self.connector_name).connection

    def handle_update(self, update):
        """Handle incoming update."""
        self.update_cache.append(update)

    def flush(self):
        """Apply the updates."""
        self.log.log(INFO_FINE, "Sorting %d updates", len(self.update_cache))
        self.update_cache.sort()

        self.log.log(INFO_FINE, "Applying %d updates", len(self.update_cache))
        con = self.connection()
        with con:
            for update in self.update_cache:
                update.apply(self)

        self.update_cache.clear()

    def execute(self, sql, params=None):
        """Execute the given sql.

        Parameters
        ----------
        sql : str
            SQL statement to execute
        params : tuple or None
            Optional parameters of the sql statement.
        """
        con = self.connection()
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
        table : str
            Table to insert into
        *params : tuple
            Values to insert into table
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

    def insert_or_ignore(self, table, *params):
        """Execute an insert or ignore statement.

        Parameters
        ----------
        table : str
            Table to insert into.
        *params : tuple
            Values to insert into table.
        """
        if table in self._insert_or_ignore_sql_cache:
            sql = self._insert_sql_cache[table]
        else:
            sql = "insert or ignore into %s.%s values (%s)"
            marks = ["?"] * len(params)
            marks = ",".join(marks)
            sql = sql % (self.store_name, table, marks)
            self._insert_sql_cache[table] = sql

        return self.execute(sql, params)
