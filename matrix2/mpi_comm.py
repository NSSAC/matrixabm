"""MPI Communication Interface."""

import pickle
import logging
import queue
import threading
from collections import defaultdict
from unittest.mock import sentinel

from mpi4py import MPI
from more_itertools import chunked

COMM_WORLD = MPI.COMM_WORLD
HOSTNAME = MPI.Get_processor_name()
WORLD_RANK = COMM_WORLD.Get_rank()
WORLD_SIZE = COMM_WORLD.Get_size()
MASTER_RANK = 0

RECV_BUFFER_SIZE = 67108864  # 64MB

SEND_BUFFER_SIZE = 100
PENDING_SEND_CLEANUP_AFTER = WORLD_SIZE

log = logging.getLogger(__name__)

# Message used to stop AsyncReceiver
StopAsyncReceiver = sentinel.StopAsyncReceiver


def is_master():
    """Return if current process is the master process."""
    return WORLD_RANK == MASTER_RANK


def pickle_dumps_adaptive(objs):
    """Make sure that the dumped pickles are within BUFFER_SIZE."""
    n = len(objs)
    assert n >= 1

    while True:
        pkls = [pickle.dumps(group) for group in chunked(objs, n)]
        max_size = max(map(len, pkls))
        if max_size < RECV_BUFFER_SIZE:
            return pkls

        log.warning("%d objects don't fit into %d bytes", n, RECV_BUFFER_SIZE)
        if n == 1:
            raise RuntimeError("Cant fit data into buffer")

        n = int(n / 2)


class AsyncSender:
    """Manager for sending messages."""

    def __init__(self):
        """Initialize."""
        self._buffer = {rank: [] for rank in range(WORLD_SIZE)}
        self._pending_sends = []

    def send(self, to, msg):
        """Send a messge."""
        self._buffer[to].append(msg)

        if len(self._buffer[to]) == SEND_BUFFER_SIZE:
            self._do_send(to)
            self._cleanup_finished_sends()

    def flush(self):
        """Flush out message buffers."""
        for rank in range(WORLD_SIZE):
            self._do_send(rank)
        self._wait_pending_sends()

    def _do_send(self, to):
        """Send all messages that have been cached."""
        if not self._buffer[to]:
            return

        pkls = pickle_dumps_adaptive(self._buffer[to])
        for pkl in pkls:
            if __debug__:
                log.debug("Sending %d bytes to %d", len(pkl), to)
            req = COMM_WORLD.Isend([pkl, MPI.CHAR], dest=to)
            self._pending_sends.append(req)

        self._buffer[to].clear()

    def _cleanup_finished_sends(self):
        """Cleanup send requests that have already completed."""
        if not self._pending_sends:
            return

        if len(self._pending_sends) < PENDING_SEND_CLEANUP_AFTER:
            return

        indices = MPI.Request.Waitsome(self._pending_sends)
        if indices is None:
            return
        indices = set(indices)

        self._pending_sends = [
            r for i, r in enumerate(self._pending_sends) if i not in indices
        ]

    def _wait_pending_sends(self):
        """Wait for all pending send requests to finish."""
        if not self._pending_sends:
            return

        MPI.Request.Waitall(self._pending_sends)
        self._pending_sends.clear()


class AsyncReceiver:
    """Manager for receiving messages."""

    def __init__(self):
        """Initialize."""
        self._msgq = queue.Queue()
        self._buf = bytearray(RECV_BUFFER_SIZE)

        self._receiver_thread = threading.Thread(target=self._keep_receiving)
        self._receiver_thread.start()

    def get(self, block=True):
        """Receive a message."""
        return self._msgq.get(block=block)

    def join(self):
        """Wait for the receiver thread to end."""
        self._receiver_thread.join()

    def _keep_receiving(self):
        """Code for the receiver thread."""
        stop_receiver = False
        while not stop_receiver:
            COMM_WORLD.Irecv([self._buf, MPI.CHAR]).wait()
            messages = pickle.loads(self._buf)
            if __debug__:
                log.debug("Received %d messages", len(messages))
            for message in messages:
                if message is StopAsyncReceiver:
                    stop_receiver = True
                    continue

                self._msgq.put(message)


class Communicator:
    """Communicate with other processes."""

    def __init__(self):
        """Initialize."""
        self._sender = AsyncSender()
        self._receiver = AsyncReceiver()

        self.send = self._sender.send
        self.flush = self._sender.flush

        self.get = self._receiver.get
        self.join = self._receiver.join

        self.hostnames = None
        self.hostname_ranks = None

        self._get_hostname_ranks()

    def finish(self):
        """Flush the sender and wait for receiver thread to finish."""
        self._sender.send(WORLD_RANK, StopAsyncReceiver)
        self._sender.flush()
        self._receiver.join()

        if __debug__:
            qsize = self._receiver._msgq.qsize()  # pylint: disable=protected-access
            if qsize:
                log.warning(
                    "Communicator finished with %d messages still in receiver queue",
                    qsize,
                )

    def get_rank(self, i, j):
        """Get the rank of the jth process on the ith node."""
        i = int(i) % len(self.hostnames)
        hostname = self.hostnames[i]
        ranks = self.hostname_ranks[hostname]

        j = int(j) % len(ranks)
        rank = ranks[j]

        return rank

    def _get_hostname_ranks(self):
        """Get the processor names for all ranks."""
        rank_hostnames = COMM_WORLD.allgather(HOSTNAME)

        hostname_ranks = defaultdict(list)
        for rank, hostname in enumerate(rank_hostnames):
            hostname_ranks[hostname].append(rank)

        hostnames = sorted(hostname_ranks)

        self.hostnames = hostnames
        self.hostname_ranks = hostname_ranks
