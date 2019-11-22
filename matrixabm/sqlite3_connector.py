"""SQLite3 connector.

The SQLite3 connector actor is used to
setup the processwise sqlite3 connection.
There should be SQLite3 connector actor on every rank.
"""

import sqlite3

import xactor.mpi_actor as asys

from . import INFO_FINE

# Shared sqlite3 connection
_SQLITE3_CONNECTION = None

LOG = asys.getLogger(__name__)


class SQLite3Connector:
    """SQLtie3 connector.

    Receives
    --------
        connect from main
        close from main
    """

    def __init__(self, store_names, dsns):
        """Initialize."""
        assert len(store_names) == len(dsns)

        self.store_names = store_names
        self.dsns = dsns

    @staticmethod
    def connection():
        """Return the shared sqlite3 connection.

        Returns
        -------
            Sqlite3 connection object
        """
        return _SQLITE3_CONNECTION

    def connect(self):
        """Setup the global sqlite3 connection.

        Sender
        ------
            main
        """
        global _SQLITE3_CONNECTION  # pylint: disable=global-statement
        if _SQLITE3_CONNECTION is not None:
            raise RuntimeError("SQLite3 connection has already been setup.")

        con = sqlite3.connect(":memory:")
        for store_name, dsn in zip(self.store_names, self.dsns):
            LOG.log(INFO_FINE, "Attaching '%s' to %s", dsn, store_name)
            sql = f"attach database ? as {store_name}"
            con.execute(sql, (dsn,))

        _SQLITE3_CONNECTION = con

    def close(self):
        """Close the global sqlite3 connection.

        Sender
        ------
            main
        """
        global _SQLITE3_CONNECTION  # pylint: disable=global-statement
        if _SQLITE3_CONNECTION is None:
            return

        LOG.log(INFO_FINE, "Closing")
        _SQLITE3_CONNECTION.close()
        _SQLITE3_CONNECTION = None
