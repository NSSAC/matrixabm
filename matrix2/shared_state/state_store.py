"""Interface for State Store implementations."""

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field

import xactor.mpi_actor as xa


@dataclass(init=False, order=True)
class StateUpdate:
    """An update message."""

    store_name: str
    order_key: str

    method: str = field(compare=False)
    args: list = field(compare=False)
    kwargs: dict = field(compare=False)

    def __init__(self, order_key, method, *args, **kwargs):
        """Construct the message."""
        self.order_key = order_key
        self.method = method
        self.args = args
        self.kwargs = kwargs

    def apply(self, store):
        """Apply the update to the store."""
        method = getattr(store, self.method)
        method(*self.args, **self.kwargs)


class StateStore(ABC):
    """State store interface."""

    def __init__(self, store_name):
        """Initialize."""
        self.store_name = store_name
        self.num_steppers_done = 0

        self.log = logging.getLogger(
            "%s(%s.%d)" % (self.__class__.__name__, self.store_name, xa.WORLD_RANK)
        )

    def agent_stepper_done(self, rank):
        """Handle agent stepper completion."""
        self.log.info("Agent stepper on rank %d is done.", rank)

        self.num_steppers_done += 1
        if self.num_steppers_done == xa.WORLD_SIZE:
            self.log.info("All agent steppers are done; flushing.")
            self.flush()
            self.num_steppers_done = 0

    @abstractmethod
    def handle_update(self, update):
        """Handle incoming update.

        Parameters
        ----------
            update: StateUpdate
                A state update
        """

    @abstractmethod
    def flush(self):
        """Apply the cached updates to the state store.

        Also sends "store_flush_done" message to MAIN
        to inform that agents are now safe to access the underlying store object.
        """
