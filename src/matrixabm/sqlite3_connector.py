"""SQLite3 connector.

The SQLite3 connector is an actor
that is used to manage sqlite3 connections.
"""

import sqlite3

import xactor as asys

from . import INFO_FINE

LOG = asys.getLogger(__name__)


class SQLite3Connector:
    """SQLtie3 connector.

    Attributes
    ----------
    store_names : list of str
        List of database names to use
    dsns : list of str
        List of sqlite3 paths corresponding the database names
    connection : sqlite3.Connection
        The sqlite3 connection object

    Receives
    --------
    * connect from Main
    * close from Main
    """

    def __init__(self, store_names, dsns):
        """Initialize."""
        assert len(store_names) == len(dsns)

        self.store_names = store_names
        self.dsns = dsns
        self.connection = None

    def connect(self):
        """Setup the sqlite3 connection.

        Sender
        ------
        Main
        """
        if self.connection is not None:
            raise RuntimeError("SQLite3 connection has already been setup.")

        con = sqlite3.connect(":memory:")
        for store_name, dsn in zip(self.store_names, self.dsns):
            LOG.log(INFO_FINE, "Attaching '%s' to %s", dsn, store_name)
            sql = f"attach database ? as {store_name}"
            con.execute(sql, (dsn,))

        self.connection = con

    def close(self):
        """Close the sqlite3 connection.

        Sender
        ------
        Main
        """
        if self.connection is None:
            return

        LOG.log(INFO_FINE, "Closing")
        self.connection.close()
        self.connection = None
