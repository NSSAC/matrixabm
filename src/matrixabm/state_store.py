"""State store interface.

The shared state of a Matrix simulation is stored in state store objects.
A Matrix simulation can have one or more state store objects.
Every state store object has a unique name.
Every state store object is replicated on every compute node running the simulation.
A state store actor manages a state store object.
Thus there is a state store actor on every compute node
for every state store object.

A state store actor receives `handle_update` messages
from the agent runner actors.
On receiving an update via a `handle_update` message
the state store actor is supposed to "cache" the update.
Once an agent runner actor is finished
for the current timestep,
it sends the state store the `handle_update_done` message.
Once the store receives all `handle_update_done` messages,
from all the agent runner actors
it executes its `flush` method.
During the flush,
all the cached updates are to be "applied"
to the underlying state store object.

Once the cached updates are flushed,
the store informs the Simulator that it is done.
"""

from time import perf_counter
from abc import ABC, abstractmethod

import xactor as asys

from . import INFO_FINE, WORLD_SIZE


class StateStore(ABC):
    """State store interface.

    Receives
    --------
    * `handle_update*` from Runner
    * `handle_update_done` from Runner

    Sends
    -----
    * `store_flush_done` to Simulator
    """

    def __init__(self, store_name, simulator_aid):
        """Initialize.

        Parameters
        ----------
        store_name : str
            Name of the current state store
        simulator_aid : str
            Proxy of the simulator actor
        """
        self.store_name = store_name
        self.simulator_proxy = asys.ActorProxy(asys.MASTER_RANK, simulator_aid)

        logger_name = "%s.%s" % (self.__class__.__name__, self.store_name)
        self.log = asys.getLogger(logger_name)

        self.num_handle_update_done = 0

    @abstractmethod
    def handle_update(self, update):
        """Handle incoming update.

        Parameters
        ----------
        update : StateUpdate
            A state update
        """

    def handle_update_done(self, rank):
        """Respond to `handle_update_done` message from a agent runner.

        Parameters
        ----------
        rank : int
            Rank of the runner
        """
        assert self.num_handle_update_done < WORLD_SIZE
        if __debug__:
            self.log.debug("Received all updates from rand %d", rank)

        self.num_handle_update_done += 1
        self._try_flush()

    def _try_flush(self):
        """Apply the cached updates to the state store."""
        self.log.log(
            INFO_FINE,
            "Can flush? (NHUD=%d/%d)",
            self.num_handle_update_done,
            WORLD_SIZE,
        )
        if self.num_handle_update_done < WORLD_SIZE:
            return

        start_time = perf_counter()
        self.flush()
        flush_time = perf_counter() - start_time

        self.simulator_proxy.store_flush_done(
            self.store_name, asys.current_rank(), flush_time,
        )

        self.num_handle_update_done = 0

    @abstractmethod
    def flush(self):
        """Apply the received updates to state store."""


class SQLite3Store(StateStore):
    """SQLite3 database file backed state store."""

    def __init__(self, store_name, simulator_proxy, sqlite3_aid):
        """Initialize.

        Parameters
        ----------
        store_name : str
            Name of the current state store
        simulator_proxy : ActorProxy
            Proxy of the simulator actor
        sqlite3_aid : str
            ID of the local SQLite3 connection manager
        """
        super().__init__(store_name, simulator_proxy)

        self.sqlite3_aid = sqlite3_aid
        self.insert_sql_cache = {}
        self.insert_or_ignore_sql_cache = {}
        self.update_cache = []

    def connection(self):
        """Get the connection from the local SQLite3 manager object."""
        return asys.local_actor(self.sqlite3_aid).connection

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
        if table in self.insert_sql_cache:
            sql = self.insert_sql_cache[table]
        else:
            sql = "insert into %s.%s values (%s)"
            marks = ["?"] * len(params)
            marks = ",".join(marks)
            sql = sql % (self.store_name, table, marks)
            self.insert_sql_cache[table] = sql

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
        if table in self.insert_or_ignore_sql_cache:
            sql = self.insert_sql_cache[table]
        else:
            sql = "insert or ignore into %s.%s values (%s)"
            marks = ["?"] * len(params)
            marks = ",".join(marks)
            sql = sql % (self.store_name, table, marks)
            self.insert_sql_cache[table] = sql

        return self.execute(sql, params)
