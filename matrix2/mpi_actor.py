"""MPI Actors.

Provides a rudimentary classical Actor model implementation on top of MPI.
"""

__all__ = ["Message", "send", "barrier", "start", "stop", "get_nodes", "get_node_ranks"]

import logging
from dataclasses import dataclass, field
from collections import defaultdict

from mpi4py import MPI

from .mpi_acomm import AsyncCommunicator

COMM_WORLD = MPI.COMM_WORLD
HOSTNAME = MPI.Get_processor_name()
WORLD_RANK = COMM_WORLD.Get_rank()
WORLD_SIZE = COMM_WORLD.Get_size()
MASTER_RANK = 0

RANK_AID_FMT = "rank-%d"
MAIN_AID = "main"

log = logging.getLogger("%s.%d" % (__name__, WORLD_RANK))


@dataclass
class Message:
    """A Message."""

    actor_id: object
    method: str
    args: list = field(default_factory=list)
    kwargs: dict = field(default_factory=dict)


class NodeRanks:
    """Get the rank of the jth process on the ith node."""

    def __init__(self):
        """Initialize."""
        rank_nodes = COMM_WORLD.allgather(HOSTNAME)

        node_ranks = defaultdict(list)
        for rank, hostname in enumerate(rank_nodes):
            node_ranks[hostname].append(rank)
        nodes = sorted(node_ranks)

        self.nodes = nodes
        self.node_ranks = dict(node_ranks)

    def get_nodes(self):
        """Return the nodes running xactor.

        Returns
        -------
            nodes: list of node names
        """
        return self.nodes

    def get_node_ranks(self, node):
        """Return the ranks on the currnet node.

        Parameters
        ----------
            node: a node name

        Returns
        -------
            ranks: list of ranks running on the given node.
        """
        return self.node_ranks[node]


class MPIProcess:
    """MPI Process Actor."""

    def __init__(self):
        self.acomm = AsyncCommunicator()
        self.local_actors = {RANK_AID_FMT % WORLD_RANK: self}

        self.stopping = False

    def _loop(self):
        """Loop through messages."""
        log.info("Starting rank loop with %d actors", len(self.local_actors))

        while not self.stopping:
            message = self.acomm.recv()
            if message.actor_id not in self.local_actors:
                log.error("Message received for non-local actor: %r", message)

            actor = self.local_actors[message.actor_id]
            try:
                method = getattr(actor, message.method)
            except AttributeError:
                log.exception(
                    "Target actor doesn't have requested method: %r, %r", actor, message
                )
                raise

            try:
                method(*message.args, **message.kwargs)
            except Exception:  # pylint: disable=broad-except
                log.exception(
                    "Exception occured while processing message: %r, %r", actor, message
                )
                raise

    def stop(self):
        """Stop the event loop after processing the current message."""
        log.info("Received stop message")

        self.acomm.finish()
        self.stopping = True

    def create_actor(self, actor_id, cls, args=None, kwargs=None):
        """Create a local actor.

        Parameters
        ----------
            actor_id: identifier for the new actor
            cls: Class used to instantiate the new actor
            args: Positional arguments for the constructor
            kwargs: Keyword arguments for the constructor
        """
        if actor_id in self.local_actors:
            raise RuntimeError("Actor with ID %s already exists" % actor_id)

        if args is None:
            args = []
        if kwargs is None:
            kwargs = {}

        actor = cls(*args, **kwargs)
        self.local_actors[actor_id] = actor


# Singleton per process MPIProcess object
_MPI_PROCESS = MPIProcess()


def send(rank, message, flush=True):
    """Send a message to the given rank.

    Parameters
    ----------
        rank: The rank to send the message to.
        message: The message to send to the rank.
    """
    if __debug__:
        assert isinstance(message.args, (list, tuple))
        assert isinstance(message.kwargs, dict)

    _MPI_PROCESS.acomm.send(rank, message)
    if flush:
        _MPI_PROCESS.acomm.flush()


def start(cls, *args, **kwargs):
    """Start the main loop.

    Sends message to the MASTER_RANK to create an with the given class,
    and run its `main' method.

    Parameters
    ----------
        cls: The main class, instantiated and its `main' method executed on MASTER_RANK
        *arg: Positional arguments for the class
        **kwargs: Keyword arguments for the class

    Returns
    -------
        Doesn't return
    """
    if WORLD_RANK == MASTER_RANK:
        rank_actor_id = RANK_AID_FMT % MASTER_RANK

        msg = Message(
            actor_id=rank_actor_id,
            method="create_actor",
            kwargs={"actor_id": MAIN_AID, "cls": cls, "args": args, "kwargs": kwargs},
        )
        _MPI_PROCESS.acomm.send(MASTER_RANK, msg)

        msg = Message(actor_id=MAIN_AID, method="main")
        _MPI_PROCESS.acomm.send(MASTER_RANK, msg)

        _MPI_PROCESS.acomm.flush()

    _MPI_PROCESS._loop()  # pylint: disable=protected-access


def stop():
    """Stop the main loop.

    Sends the stop message to all the ranks.
    """
    for rank in range(WORLD_SIZE):
        rank_actor_id = RANK_AID_FMT % rank
        msg = Message(actor_id=rank_actor_id, method="stop")
        _MPI_PROCESS.acomm.send(rank, msg)

    _MPI_PROCESS.acomm.flush()


def barrier():
    """Perform a barrier synchornization."""
    COMM_WORLD.Barrier()


# Class with the information about the nodes and ranks running on them
_NODE_RANKS = NodeRanks()

get_nodes = _NODE_RANKS.get_nodes

get_node_ranks = _NODE_RANKS.get_node_ranks
