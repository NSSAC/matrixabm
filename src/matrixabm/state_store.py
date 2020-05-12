"""State store interface.

The shared state of a Matrix simulation is stored in state store objects.
A Matrix simulation can have one or more state store objects.
Every state store object has a unique name.
Every state store object is replicated on every compute node
running the simulation.
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
"""

from time import time
from abc import ABC, abstractmethod

import xactor as asys

from . import INFO_FINE
from .standard_actors import MAIN

WORLD_SIZE = len(asys.ranks())


class StateStore(ABC):
    """State store interface.

    Receives
    --------
    * handle_update* from Runner
    * handle_update_done from Runner

    Sends
    -----
    * store_flush_done to Simulator
    """

    def __init__(self, store_name):
        """Initialize."""
        self.store_name = store_name
        self.log = asys.getLogger("%s.%s" % (self.__class__.__name__, self.store_name))

        self.num_handle_update_done = 0

    @abstractmethod
    def handle_update(self, update):
        """Handle incoming update.

        Sender
        ------
        Runner


        Parameters
        ----------
        update : StateUpdate
            A state update
        """

    def handle_update_done(self, rank):
        """Respond to `handle_update_done` message from a agent runner.

        Sender
        ------
        Runner

        Parameters
        ----------
        rank : int
            Rank of the runner
        """
        assert self.num_handle_update_done < WORLD_SIZE
        if __debug__:
            self.log.debug("Agent runner on %d is done", rank)

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

        start_time = time()
        self.flush()
        flush_time = time() - start_time

        MAIN.store_flush_done(
            self.store_name, asys.current_rank(), flush_time,
        )

        self.num_handle_update_done = 0

    @abstractmethod
    def flush(self):
        """Apply the received updates to state store."""
