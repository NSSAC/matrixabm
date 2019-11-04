"""Greedy Agent Coordinator Implementation."""

import enum
import logging

import xactor.mpi_actor as xa

from . import AID_RUNNER, AID_MAIN

class AgentCoordinatorState(enum.IntEnum):
    """States of the agent coordinator."""

    WAITING_FOR_STEP_START = enum.auto()
    WAITING_FOR_AGENT_CREATION_DONE = enum.auto()
    BALANCING = enum.auto()
    COORDINATING = enum.auto()
    WAITING_FOR_STEP_FINISH = enum.auto()


class AgentCoordinator:
    """Agent coordinator implementation."""

    def __init__(self, balancer, n_ranks):
        """Initialize.

        Parameters
        ----------
            n_ranks: int
                Number of ranks
        """
        self.state = AgentCoordinatorState.WAITING_FOR_STEP_START

        self.n_ranks = n_ranks
        self.balancer = balancer

        self.agent_constructor = dict()
        self.num_agents_created = 0
        self.num_agents_died = 0
        self.num_runners_done = 0

        self.log = logging.getLogger(
            "%s(%d)" % (self.__class__.__name__, xa.WORLD_RANK)
        )

    def step_started(self):
        """Log that the step has started."""
        assert self.state == AgentCoordinatorState.WAITING_FOR_STEP_START
        self.state = AgentCoordinatorState.WAITING_FOR_AGENT_CREATION_DONE

    def agent_created(self, agent_id, constructor, step_time, memory_usage):
        """Log an agent creation.

        Parameters
        ----------
            agent_id: string
                ID of the dead agent.
            constructor: object
                Constructor used to create the agent
            step_time: float
                Initial estimate of seconds used by the agent to execute per unit simulated real time
            memory_usage: float
                Initial estimate of the memory usage of the agent in bytes
        """
        assert self.state == AgentCoordinatorState.WAITING_FOR_AGENT_CREATION_DONE

        self.agent_constructor[agent_id] = constructor
        self.balancer.add_object(agent_id, memory_usage, step_time)
        self.num_agents_created += 1

    def agent_creation_done(self):
        """Log that agent creation is done."""
        assert self.state == AgentCoordinatorState.WAITING_FOR_AGENT_CREATION_DONE
        self.state = AgentCoordinatorState.BALANCING

        self.balancer.balance()

        self.state = AgentCoordinatorState.COORDINATING

        for agent_id in self.balancer.new_objects:
            constructor = self.agent_constructor[agent_id]
            msg = xa.Message("create_agent", agent_id, constructor)
            rank = self.balancer.object_bucket[agent_id]
            xa.send(rank, AID_RUNNER, msg, immediate=False)

        for agent_id in self.balancer.object_bucket_prev:
            src = self.balancer.object_bucket_prev[agent_id]
            dst = self.balancer.object_bucket[agent_id]
            msg = xa.Message("move_agent", agent_id, src, dst)
            xa.send(src, AID_RUNNER, msg, immediate=False)

        msg = xa.Message("agent_coordination_done")
        xa.send(xa.EVERY_RANK, AID_RUNNER, msg, immediate=False)

        xa.flush()

        self.state = AgentCoordinatorState.WAITING_FOR_STEP_FINISH

    def agent_step_profile(self, agent_id, step_time, memory_usage, is_alive, timestep):
        """Log an agent step profile.

        Parameters
        ----------
            agent_id: string
                ID of the dead agent.
            step_time: float
                Seconds used by the agent to execute the current timestep
            memory_usage: float
                Memory usage of the agent in bytes
            timestep: Timestep
                The current (just finished) timestep
        """
        assert self.state == AgentCoordinatorState.WAITING_FOR_STEP_FINISH

        if __debug__:
            self.log.debug(
                "Agent step profile: %s: (%gs %gKb alive=%s) %s",
                agent_id,
                step_time,
                memory_usage // 1024,
                is_alive,
                timestep,
            )

        if not is_alive:
            self.balancer.delete_object(agent_id)
            self.num_agents_died += 1
            return

        scaled_step_time = step_time / (timestep.end - timestep.start)
        self.balancer.update_load(agent_id, memory_usage, scaled_step_time)

    def runner_done(self):
        """Log that a runner has completed the step."""
        assert self.state == AgentCoordinatorState.WAITING_FOR_STEP_FINISH

        self.num_runners_done += 1
        if self.num_runners_done < self.n_ranks:
            return

        msg = xa.Message("coordinator_done")
        xa.send(xa.MASTER_RANK, AID_MAIN, msg, immediate=True)

        self.num_runners_done = 0
        self.balancer.reset()

        self.state = AgentCoordinatorState.WAITING_FOR_STEP_START
