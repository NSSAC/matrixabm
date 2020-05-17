"""Resource manager actors.

This module defines actors whose sole task is to
manage a resouce and make them available to other actors.
"""

import sqlite3

import xactor as asys
from tensorboardX import SummaryWriter

from . import INFO_FINE

LOG = asys.getLogger(__name__)


class SQLite3Manager:
    """SQLtie3 manager.

    The SQLite3 manager manages sqlite3 connections.

    Attributes
    ----------
    dbnames : list of str
        List of database names to use
    dsns : list of str
        List of sqlite3 paths corresponding the database names
    connection : sqlite3.Connection
        The sqlite3 connection object
    """

    def __init__(self, dbnames, dsns):
        """Initialize."""
        assert len(dbnames) == len(dsns)

        self.dbnames = dbnames
        self.dsns = dsns

        con = sqlite3.connect(":memory:")
        for dbname, dsn in zip(self.dbnames, self.dsns):
            LOG.log(INFO_FINE, "Attaching '%s' to %s", dsn, dbname)
            sql = f"attach database ? as {dbname}"
            con.execute(sql, (dsn,))

        self.connection = con

    def __del__(self):
        self.close()

    def close(self):
        """Close the sqlite3 connection."""
        if self.connection is None:
            return

        LOG.log(INFO_FINE, "Closing sqlite3 connection; %r", self.dsns)
        self.connection.close()
        self.connection = None


class TensorboardWriter:
    """Tensorboard Summary Writer.

    This actor is used to
    manage a tensorboard SummaryWriter object.

    Attributes
    ----------
    summary_dir : str
        The summary directory
    summary_writer : tensorboardX.SummaryWriter
        The underlying summary writer
    """

    def __init__(self, summary_dir):
        self.summary_dir = summary_dir

        LOG.log(INFO_FINE, "Creating tensorboard summary writer for '%s'", summary_dir)
        self.summary_writer = SummaryWriter(str(summary_dir))

        self.add_scalar = self.summary_writer.add_scalar
        self.add_histogram = self.summary_writer.add_histogram

        self.flush = self.summary_writer.flush

    def __del__(self):
        self.close()

    def close(self):
        """Close the summary writer."""
        if self.summary_writer is None:
            return

        LOG.log(INFO_FINE, "Closing summary writer; %r", self.summary_dir)
        self.summary_writer.close()
        self.summary_writer = None
