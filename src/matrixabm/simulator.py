"""The Main Simulator.

The simulator is responsible for coordinating the overall simulation.
It creates the actors and sends messages to them to start the rounds.
"""

from abc import ABC, abstractmethod
from time import time

import xactor as asys

from . import INFO_FINE
# from .summary_writer import get_summary_writer
from .standard_actors import AID_POPULATION, AID_COORDINATOR, AID_RUNNER
from .standard_actors import POPULATION, COORDINATOR, EVERY_RUNNER
from .coordinator import Coordinator
from .runner import Runner

LOG = asys.getLogger(__name__)
WORLD_SIZE = len(asys.ranks())


class Simulator(ABC):
    """The main simulator.

    Receives
    --------
    * store_flush_done from StateStore(s)
    * coordinator_done from Coordinator

    Sends
    -----
    * step to Coordinator
    * step to Runner
    * create_agents to Population
    """

    def __init__(self, summary_writer_aname):
        """Initialize."""
        self.summary_writer_aname = summary_writer_aname

        self.stores = []
        self.timestep_generator = None

        self.timestep = None
        self.round_start_time = None
        self.round_end_time = None

        # Step variables
        self.flag_coordinator_done = None
        self.num_store_flush_done = None
        self.store_rank_flush_time = None

        self._prepare_for_next_step()

    def _prepare_for_next_step(self):
        """Prepare for next step."""
        self.flag_coordinator_done = False
        self.num_store_flush_done = {store_name: 0 for store_name in self.stores}
        self.store_rank_flush_time = {}

    def _write_summary(self):
        """Log the summary of activities."""
        summary_writer = asys.local_actor(self.summary_writer_aname)
        if summary_writer is None:
            return

        round_time = self.round_end_time - self.round_start_time
        summary_writer.add_scalar("round_time", round_time, self.timestep.step)

        for (store_name, rank), flush_time in self.store_rank_flush_time.items():
            summary_writer.add_scalar(
                f"store_flush_time/{store_name}/{rank}", flush_time, self.timestep.step
            )

        summary_writer.flush()

    def _try_start_step(self, starting):
        if not starting:
            n_nodes = len(asys.nodes())
            nsfd_str = [
                (name, num, n_nodes) for name, num in self.num_store_flush_done.items()
            ]
            nsfd_str = ["%s=%d/%d" % tup for tup in nsfd_str]
            nsfd_str = ",".join(nsfd_str)
            LOG.log(
                INFO_FINE,
                "Can start step? (FCD=%s,%s)",
                self.flag_coordinator_done,
                nsfd_str,
            )
            if not self.flag_coordinator_done:
                return
            for store_name in self.stores:
                if self.num_store_flush_done[store_name] < n_nodes:
                    return

        if not starting:
            self.round_end_time = time()
            self._write_summary()

        self.timestep = self.timestep_generator.get_next_timestep()
        if self.timestep is None:
            LOG.info("Simulation finished.")
            asys.stop()
            return
        self.round_start_time = time()
        self.round_end_time = None

        LOG.info("Starting timestep %f", self.timestep.step)
        POPULATION.create_agents(self.timestep)
        COORDINATOR.step(self.timestep)
        EVERY_RUNNER.step(self.timestep)
        asys.flush()

        self._prepare_for_next_step()

    def main(self):
        """Start the simulation."""
        # Create the timestep generator
        self.timestep_generator = self.TimestepGenerator().construct()

        # Create the population actor
        ctor = self.Population()
        asys.create_actor(
            asys.MASTER_RANK, AID_POPULATION, ctor.cls, *ctor.args, **ctor.kwargs
        )

        # Create the coordinator
        balancer = self.LoadBalancer().construct()
        asys.create_actor(asys.MASTER_RANK, AID_COORDINATOR, Coordinator, balancer, self.summary_writer_aname)

        # Create the state stores
        store_proxies = {}
        for i, (store_name, ctor) in enumerate(self.StateStores()):
            # Figure out where to place the stores
            store_ranks = []
            for node in asys.nodes():
                ranks = asys.node_ranks(node)
                rank = ranks[i % len(ranks)]
                store_ranks.append(rank)

            # Create the stores on those ranks
            for rank in store_ranks:
                asys.create_actor(rank, store_name, ctor.cls, *ctor.args, **ctor.kwargs)

            # Create the store proxies
            store_proxies[store_name] = asys.ActorProxy(store_ranks, store_name)

            # Note the store
            self.stores.append(store_name)

        # Create the runners
        for rank in asys.ranks():
            asys.create_actor(rank, AID_RUNNER, Runner, store_proxies)

        self._try_start_step(starting=True)

    def store_flush_done(self, store_name, rank, flush_time):
        """Log that the store flush for the given store was completed.

        Sender
        ------
        StateStore(s)

        Parameters
        ----------
        store_name: str
            Name of the state store
        rank: int
            The rank on which the store was running
        flush_time:
            Number of seconds taken by the flush operation.
        """
        if __debug__:
            LOG.debug("The store %s on rank %d has completed flush", store_name, rank)

        assert self.num_store_flush_done[store_name] < len(asys.nodes())

        self.num_store_flush_done[store_name] += 1
        self.store_rank_flush_time[store_name, rank] = flush_time
        self._try_start_step(starting=False)

    def coordinator_done(self):
        """Log that the coordinator has finished.

        Sender
        ------
        Coordinator
        """
        if __debug__:
            LOG.debug("The coordinator is done with the current round")

        assert not self.flag_coordinator_done

        self.flag_coordinator_done = True
        self._try_start_step(starting=False)

    @abstractmethod
    def TimestepGenerator(self):
        """Get the timestep generator constructor.

        Returns
        -------
        constructor: Constructor
            Constructor of the timestep generator class
        """

    @abstractmethod
    def Population(self):
        """Get the population constructor.

        Returns
        -------
        constructor: Constructor
            Constructor of the population class
        """

    @abstractmethod
    def StateStores(self):
        """Get the state store constructors.

        Returns
        -------
        List of 2 tuples [(store_name, constructor)]
            store_name: str
                Name of the state store
            constructor: Constructor
                Constructor of the state store class
        """

    @abstractmethod
    def LoadBalancer(self):
        """Get the load balancer constructor.

        Returns
        -------
        constructor: LoadBalancer
            Return an instance of the load balancer
        """
