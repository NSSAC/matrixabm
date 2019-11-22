"""Agent coordinator.

The agent coordinator is responsible for
assigning agents to runners on different ranks
and making sure the overall load is balanced.
There is a single coordinator actor in every simulation.
"""

from time import time
import numpy as np

import xactor.mpi_actor as asys

from . import INFO_FINE
from .summary_writer import get_summary_writer
from .standard_actors import RUNNERS, EVERY_RUNNER, MAIN

LOG = asys.getLogger(__name__)


class Coordinator:
    """Agent coordinator.

    Receives
    --------
        step from main
        create_agent* from population
        create_agent_done from population
        agent_step_profile* from runner(rank)
        agent_step_profile_done from runner(rank)

    Sends
    -----
        create_agent* to runner(rank)
        create_agent_done to runner(rank)
        move_agent* to runner(rank)
        move_agent_done to runner(rank)
        coordinator_done to main
    """

    def __init__(self, balancer):
        """Initialize.

        Parameters
        ----------
            balancer: LoadBalancer
                The agent load balancer
        """
        self.balancer = balancer

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

        self.rank_step_time = [0.0] * asys.WORLD_SIZE
        self.rank_memory_usage = [0.0] * asys.WORLD_SIZE
        self.rank_n_updates = [0] * asys.WORLD_SIZE

        self.agent_step_time = {}
        self.agent_memory_usage = {}
        self.agent_n_updates = {}

        self.balancing_time = -1.0

    def _write_summary(self):
        """Log the summary of activites."""
        summary_writer = get_summary_writer()
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

        for rank in range(asys.WORLD_SIZE):
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
        EVERY_RUNNER.create_agent_done(send_immediate=True)

        for agent_id, src, dst in self.balancer.get_moving_objects():
            RUNNERS[src].move_agent(agent_id, dst)
        EVERY_RUNNER.move_agent_done(send_immediate=True)

    def _try_finish_step(self):
        """Try to finish the step."""
        LOG.log(
            INFO_FINE,
            "Can finish step? (TS=%s,CAD=%s,NASPD=%d/%d)",
            bool(self.timestep),
            self.flag_create_agent_done,
            self.num_agent_step_profile_done,
            asys.WORLD_SIZE,
        )
        if self.timestep is None:
            return
        if not self.create_agent_done:
            return
        if self.num_agent_step_profile_done < asys.WORLD_SIZE:
            return

        MAIN.coordinator_done(send_immediate=True)
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
            population

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
            population
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
            runner(rank)

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
            runner(rank)

        Parameters
        ----------
            rank: int
                Rank of the agent runner
        """
        assert self.num_agent_step_profile_done < asys.WORLD_SIZE
        if __debug__:
            LOG.debug("Runner on %d is done", rank)

        self.num_agent_step_profile_done += 1
        self._try_finish_step()
