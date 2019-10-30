"""SQLite3 shared state store."""

import xactor.mpi_actor as xa

from . import AID_MAIN
from .sqlite3_connector import get_connection
from .state_store import StateStore


class SQLite3Store(StateStore):
    """Sqlite3 state store."""

    def __init__(self, store_name):
        """Initialize.

        Parameters
        ----------
            store_name: name of the sqltie3 store
        """
        super().__init__(store_name)

        self.update_cache = []

    def handle_update(self, update):
        """Handle incoming update.

        Parameters
        ----------
            update: A state update
        """
        self.update_cache.append(update)

    def execute_sql(self, sql, params=None):
        """Execute the given sql.

        Parameters
        ----------
            sql: sql statement to execute
            params: optional parameters of the sql statement.
        """
        con = get_connection()
        if params is None:
            con.execute(sql)
        else:
            con.execute(sql, params)

    def flush(self):
        """Apply the cached updates to the state store."""
        if not self.update_cache:
            return

        self.log.info("Sorting %d updates", len(self.update_cache))
        self.update_cache.sort()

        self.log.info("Applying %d updates", len(self.update_cache))
        con = get_connection()
        with con:
            for update in self.update_cache:
                update.apply(self)

        msg = xa.Message(AID_MAIN, "store_flush_done", self.store_name, xa.WORLD_RANK)
        xa.send(xa.MASTER_RANK, msg)

        self.update_cache.clear()
        self.log.info("Cache flushed")
