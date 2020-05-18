"""The Simulator."""

from time import perf_counter

import xactor as asys

from . import INFO_FINE

LOG = asys.getLogger(__name__)


class Simulator:
    """The Simulator.

    The simulator is responsible for coordinating the overall simulation.
    It expects all the actors have already been created.

    Receives
    --------
    * `store_flush_done` from StateStore(s)
    * `coordinator_done` from Coordinator

    Sends
    -----
    * `step` to Coordinator
    * `step` to Runner
    * `create_agents` to Population
    """

    def __init__(
        self,
        coordinator_aid,
        runner_aid,
        population_aid,
        timestep_generator_aid,
        store_names,
        summary_writer_aid=None,
    ):
        """Initialize."""
        self.coordinator_proxy = asys.ActorProxy(asys.MASTER_RANK, coordinator_aid)
        self.every_runner_proxy = asys.ActorProxy(asys.EVERY_RANK, runner_aid)
        self.population_proxy = asys.ActorProxy(asys.MASTER_RANK, population_aid)

        self.timestep_generator_aid = timestep_generator_aid
        self.summary_writer_aid = summary_writer_aid
        self.store_names = store_names

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
        self.num_store_flush_done = {store_name: 0 for store_name in self.store_names}
        self.store_rank_flush_time = {}

    def _write_summary(self):
        """Log the summary of activities."""
        if self.summary_writer_aid is None:
            return
        summary_writer = asys.local_actor(self.summary_writer_aid)
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
            for store_name in self.store_names:
                if self.num_store_flush_done[store_name] < n_nodes:
                    return

        if not starting:
            self.round_end_time = perf_counter()
            self._write_summary()

        timestep_generator = asys.local_actor(self.timestep_generator_aid)
        self.timestep = timestep_generator.get_next_timestep()
        if self.timestep is None:
            LOG.info("Simulation finished.")
            asys.stop()
            return
        self.round_start_time = perf_counter()
        self.round_end_time = None

        LOG.info("Starting timestep %f", self.timestep.step)
        self.population_proxy.create_agents(self.timestep)
        self.coordinator_proxy.step(self.timestep)
        self.every_runner_proxy.step(self.timestep)

        self._prepare_for_next_step()

    def start(self):
        """Start the simulation."""
        self._try_start_step(starting=True)

    def store_flush_done(self, store_name, rank, flush_time):
        """Log that the store flush for the given store was completed.

        Parameters
        ----------
        store_name : str
            Name of the state store
        rank : int
            The rank on which the store was running
        flush_time : float
            Number of seconds taken by the flush operation.
        """
        if __debug__:
            LOG.debug("The store %s on rank %d has completed flush", store_name, rank)

        assert self.num_store_flush_done[store_name] < len(asys.nodes())

        self.num_store_flush_done[store_name] += 1
        self.store_rank_flush_time[store_name, rank] = flush_time
        self._try_start_step(starting=False)

    def coordinator_done(self):
        """Log that the coordinator has finished."""
        if __debug__:
            LOG.debug("The coordinator is done with the current round")

        assert not self.flag_coordinator_done

        self.flag_coordinator_done = True
        self._try_start_step(starting=False)
