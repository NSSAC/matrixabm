"""SQLite3 connector.

Provides the SQLite3 connector actor.
The job of the actor is to setup the process's sqltie3 connection.
"""
# pylint: disable=global-statement

import logging
import sqlite3

import xactor.mpi_actor as xa

# Shared sqlite3 connection
_sqlite3_connection = None


def get_connection():
    """Return the shared sqlite3 connection.

    Returns
    -------
        Sqlite3 connection object
    """
    return _sqlite3_connection


class SQLite3Connector:
    """Sqltie3 connector."""

    def __init__(self, store_names, dsns):
        """Initialize."""
        assert len(store_names) == len(dsns)

        self.store_names = store_names
        self.dsns = dsns

        self.log = logging.getLogger(
            "%s(%d)" % (self.__class__.__name__, xa.WORLD_RANK)
        )

    def connect(self):
        """Setup the global sqlite3 connection."""
        global _sqlite3_connection
        if _sqlite3_connection is not None:
            raise RuntimeError("SQLite3 connection has already been setup.")

        con = sqlite3.connect(":memory:")
        for store_name, dsn in zip(self.store_names, self.dsns):
            self.log.info("Attaching '%s' to %s", dsn, store_name)
            sql = f"attach database '{dsn}' as {store_name}"
            con.execute(sql)

        _sqlite3_connection = con

    def close(self):
        """Close the global sqlite3 connection."""
        global _sqlite3_connection
        if _sqlite3_connection is None:
            return

        self.log.info("Closing")
        _sqlite3_connection.close()
        _sqlite3_connection = None
