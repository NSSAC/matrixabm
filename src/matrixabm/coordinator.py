"""Agent coordinator.

The agent coordinator is responsible for
assigning agents to runners on different ranks
and making sure the overall load is balanced.
There is a single coordinator actor in every simulation.
"""

from time import time
import numpy as np

import xactor as asys

from . import INFO_FINE
# from .summary_writer import get_summary_writer
from .standard_actors import RUNNERS, EVERY_RUNNER, MAIN

LOG = asys.getLogger(__name__)
WORLD_SIZE = len(asys.ranks())


class Coordinator:
    """Agent coordinator.

    Receives
    --------
    * step from Simulator
    * create_agent* from Population
    * create_agent_done from Population
    * agent_step_profile* from Runner
    * agent_step_profile_done from Runner

    Sends
    -----
    * create_agent* to Runner
    * create_agent_done to Runner
    * move_agent* to Runner
    * move_agent_done to Runner
    * coordinator_done to Simulator
    """

    def __init__(self, balancer, summary_writer_aname):
        """Initialize.

        Parameters
        ----------
        balancer: LoadBalancer
            The agent load balancer
        summary_writer_aname : str
            The name of the local summary writer
        """
        self.balancer = balancer
        self.summary_writer_aname = summary_writer_aname

        self.num_agents_created = 0
        self.num_agents_died = 0

        # Step variables
        self.timestep = None
        self.agent_constructor = None
        self.flag_create_agent_done = None
        self.num_agent_step_profile_done = None
        self.rank_step_time = None
        self.rank_memory_usage = None
        self.rank_n_updates = None
        self.agent_step_time = None
        self.agent_memory_usage = None
        self.agent_n_updates = None
        self.balancing_time = None

        self._prepare_for_next_step()

    def _prepare_for_next_step(self):
        """Prepare for next step."""
        LOG.log(INFO_FINE, "Preparing for next step")

        self.timestep = None
        self.agent_constructor = {}
        self.balancer.reset()

        self.flag_create_agent_done = False
        self.num_agent_step_profile_done = 0

        self.rank_step_time = [0.0] * WORLD_SIZE
        self.rank_memory_usage = [0.0] * WORLD_SIZE
        self.rank_n_updates = [0] * WORLD_SIZE

        self.agent_step_time = {}
        self.agent_memory_usage = {}
        self.agent_n_updates = {}

        self.balancing_time = -1.0

    def _write_summary(self):
        """Log the summary of activites."""
        summary_writer = asys.local_actor(self.summary_writer_aname)
        if summary_writer is None:
            return

        summary_writer.add_scalar(
            "num_agents_created", self.num_agents_created, self.timestep.step
        )
        summary_writer.add_scalar(
            "num_agents_died", self.num_agents_died, self.timestep.step
        )
        summary_writer.add_scalar(
            "num_residual_population",
            self.num_agents_created - self.num_agents_died,
            self.timestep.step,
        )

        for rank in range(WORLD_SIZE):
            summary_writer.add_scalar(
                f"rank_step_time/{rank}", self.rank_step_time[rank], self.timestep.step
            )
            summary_writer.add_scalar(
                f"rank_memory_usage/{rank}",
                self.rank_memory_usage[rank],
                self.timestep.step,
            )
            summary_writer.add_scalar(
                f"rank_n_updates/{rank}", self.rank_n_updates[rank], self.timestep.step
            )

        agent_step_time = np.array(list(self.agent_step_time.values()))
        summary_writer.add_histogram(
            "agent_step_time", agent_step_time, self.timestep.step, bins="auto"
        )
        agent_memory_usage = np.array(list(self.agent_memory_usage.values()))
        summary_writer.add_histogram(
            "agent_memory_usage", agent_memory_usage, self.timestep.step, bins="auto"
        )
        agent_n_updates = np.array(list(self.agent_n_updates.values()))
        summary_writer.add_histogram(
            "agent_n_updates", agent_n_updates, self.timestep.step, bins="auto"
        )

        summary_writer.add_scalar(
            "balancing_time", self.balancing_time, self.timestep.step
        )

        summary_writer.flush()

    def _try_load_balance(self):
        """Try to start the load balancing step."""
        LOG.log(
            INFO_FINE,
            "Can balance load? (TS=%s,CAD=%s)",
            bool(self.timestep),
            self.flag_create_agent_done,
        )
        if self.timestep is None:
            return
        if not self.flag_create_agent_done:
            return

        start_time = time()
        self.balancer.balance()
        self.balancing_time = time() - start_time

        for agent_id, rank in self.balancer.get_new_objects():
            constructor = self.agent_constructor[agent_id]
            RUNNERS[rank].create_agent(agent_id, constructor)
        EVERY_RUNNER.create_agent_done()

        for agent_id, src, dst in self.balancer.get_moving_objects():
            RUNNERS[src].move_agent(agent_id, dst)
        EVERY_RUNNER.move_agent_done()

    def _try_finish_step(self):
        """Try to finish the step."""
        LOG.log(
            INFO_FINE,
            "Can finish step? (TS=%s,CAD=%s,NASPD=%d/%d)",
            bool(self.timestep),
            self.flag_create_agent_done,
            self.num_agent_step_profile_done,
            WORLD_SIZE,
        )
        if self.timestep is None:
            return
        if not self.create_agent_done:
            return
        if self.num_agent_step_profile_done < WORLD_SIZE:
            return

        MAIN.coordinator_done()
        self._write_summary()
        self._prepare_for_next_step()

    def step(self, timestep):
        """Log the start of the next timestep.

        Sender
        -----
            The main actor

        Parameters
        ----------
            timestep: The current (to start) timestep
        """
        assert self.timestep is None

        self.timestep = timestep
        self._try_load_balance()

    def create_agent(self, agent_id, constructor, step_time, memory_usage):
        """Log an agent creation.

        Sender
        ------
        Population

        Parameters
        ----------
        agent_id: str
            ID of the to be created agent
        constructor: Constructor
            Constructor of the agent
        step_time: float
            Initial estimate step_time per unit simulated real time (in seconds)
        memory_usage: float
            Initial estimate of memory usage (in bytes)
        """
        self.agent_constructor[agent_id] = constructor
        self.balancer.add_object(agent_id, memory_usage, step_time)
        self.num_agents_created += 1

    def create_agent_done(self):
        """Log that agent creation is done.

        Sender
        ------
        * Population
        """
        assert not self.flag_create_agent_done

        self.flag_create_agent_done = True
        self._try_load_balance()

    def agent_step_profile(
        self, rank, agent_id, step_time, memory_usage, n_updates, is_alive
    ):
        """Log an agent step profile.

        Sender
        ------
        Runner

        Parameters
        ----------
        rank: int
            Rank of the agent runner
        agent_id: str
            ID of the dead agent
        step_time: float
            Time taken by the agent to execute current timestep (in seconds)
        memory_usage: float
            Memory usage of the agent (in bytes)
        n_updates: int
            Number of updates produced by the agent
        is_alive: bool
            True if agent will generate events in the future
        """
        self.rank_step_time[rank] += step_time
        self.rank_memory_usage[rank] += memory_usage
        self.rank_n_updates[rank] += n_updates

        self.agent_step_time[agent_id] = step_time
        self.agent_memory_usage[agent_id] = memory_usage
        self.agent_n_updates[agent_id] = n_updates

        if not is_alive:
            self.balancer.delete_object(agent_id)
            self.num_agents_died += 1
            return

        scaled_step_time = step_time / (self.timestep.end - self.timestep.start)
        self.balancer.update_load(agent_id, memory_usage, scaled_step_time)

    def agent_step_profile_done(self, rank):
        """Log that a runner has completed the step.

        Sender
        ------
        Runner

        Parameters
        ----------
        rank : int
            Rank of the agent runner
        """
        assert self.num_agent_step_profile_done < WORLD_SIZE
        if __debug__:
            LOG.debug("Runner on %d is done", rank)

        self.num_agent_step_profile_done += 1
        self._try_finish_step()
