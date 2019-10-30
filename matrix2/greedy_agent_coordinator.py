"""Greedy Agent Coordinator Implementation."""

import heapq
import random
import logging
from collections import defaultdict

import numpy as np
import xactor.mpi_actor as xa

from . import AID_STEPPER

LAMBDA_STEP = 0.9
LAMBDA_MEM = 0.9
LAMBDA_COMPUTE = 0.9
IMBALANCE_TOLERANCE = 0.1


class GreedyAgentCoordinator:
    """Greedy agent distributor implementation."""

    def __init__(self, n_ranks):
        """Initialize.

        Parameters
        ----------
            n_ranks: int
                Number of ranks
        """
        self.n_ranks = n_ranks
        self.rank_agents = [set() for rank in range(self.n_ranks)]
        self.agent_rank = dict()

        self.agent_step_time = dict()
        self.agent_memory_usage = dict()

        self.agent_constructor = dict()

        self.num_agents_created = 0
        self.num_agents_died = 0

        self.rank_load = [0.0] * self.n_ranks
        self.agent_load = {}

        self.log = logging.getLogger(
            "%s(%d)" % (self.__class__.__name__, xa.WORLD_RANK)
        )

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
            del self.agent_step_time[agent_id]
            del self.agent_memory_usage[agent_id]

            rank = self.agent_rank[agent_id]
            self.rank_agents[rank].remove(agent_id)
            del self.agent_rank[agent_id]

            self.num_agents_died += 0
            return

        prev_step_time = self.agent_step_time[agent_id]
        step_time = step_time / (timestep.end - timestep.start)
        step_time = (1 - LAMBDA_STEP) * prev_step_time + LAMBDA_STEP * step_time
        self.agent_step_time[agent_id] = step_time

        prev_memory_usage = self.agent_memory_usage[agent_id]
        memory_usage = (1 - LAMBDA_MEM) * prev_memory_usage + LAMBDA_MEM * memory_usage
        self.agent_memory_usage[agent_id] = memory_usage

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
        if __debug__:
            self.log.debug(
                "Agent created: %s: (%gs %gKb)",
                agent_id,
                step_time,
                memory_usage // 1024,
            )

        self.agent_constructor[agent_id] = constructor

        self.agent_step_time[agent_id] = step_time
        self.agent_memory_usage[agent_id] = memory_usage

        # Do an random initial assignment
        rank = random.randint(0, self.n_ranks - 1)
        self.agent_rank[agent_id] = rank
        self.rank_agents[rank].add(agent_id)

        self.num_agents_created += 1

    def get_agent_load(self):
        """Compute agent load."""
        max_agent_compute_load = max(self.agent_step_time.values())
        max_agent_memory_usage = max(self.agent_memory_usage.values())

        agent_load = {}
        for agent_id in self.agent_rank:
            compute_load = self.agent_step_time[agent_id] / max_agent_compute_load
            memory_load = self.agent_memory_usage[agent_id] / max_agent_memory_usage
            agent_load[agent_id] = (
                LAMBDA_COMPUTE * compute_load + (1 - LAMBDA_COMPUTE) * memory_load
            )

        return agent_load

    def get_rank_load(self, agent_load):
        """Compute rank load."""
        rank_load = [0.0] * self.n_ranks
        for rank in range(self.n_ranks):
            rank_load[rank] = sum(
                agent_load[agent_id] for agent_id in self.rank_agents[rank]
            )

        return rank_load

    @staticmethod
    def get_imbalance(rank_load):
        """Check if there is a load imbalance."""
        min_load = min(rank_load)
        max_load = max(rank_load)
        sum_load = sum(rank_load)

        return (max_load - min_load) / sum_load

    def greedy_move(self, agent_load, rank_load, rank_agents, agent_rank):
        """Greedily select an agents to move from max loaded rank to min loaded rank."""
        from_rank = np.argmax(rank_load)
        to_rank = np.argmin(rank_load)

        agent_load_heap = [(agent_load[aid], aid) for aid in rank_agents[from_rank]]
        heapq.heapify(agent_load_heap)

        agent_ids = []
        while agent_load_heap:
            load, agent_id = heapq.heappop(agent_load_heap)

            # If movement will still leave the from_rank
            # more or equally loaded than to_rank
            # then move the agent
            if rank_load[from_rank] - load >= rank_load[to_rank] + load:
                agent_ids.append(agent_id)
                rank_load[from_rank] -= load
                rank_load[to_rank] += load
                rank_agents[from_rank].remove(agent_id)
                rank_agents[to_rank].add(agent_id)
                agent_rank[agent_id] = to_rank
            else:
                return agent_ids, rank_load, rank_agents, agent_rank

    def do_move_agents(self, moving_agents, new_agent_rank):
        """Send the move agent commands to steppers and update local registry."""
        rank_moving_agent_ids = defaultdict(list)
        rank_moving_dst_ranks = defaultdict(list)
        for agent_id in moving_agents:
            from_rank = self.agent_rank[agent_id]
            to_rank = new_agent_rank[agent_id]
            if from_rank == to_rank:
                continue

            rank_moving_agent_ids[from_rank].append(agent_id)
            rank_moving_dst_ranks[from_rank].append(to_rank)

            self.rank_agents[from_rank].remove(agent_id)
            self.rank_agents[to_rank].add(agent_id)
            self.agent_rank[agent_id] = to_rank

        for rank in rank_moving_agent_ids:
            agent_ids = rank_moving_agent_ids[from_rank]
            dst_ranks = rank_moving_dst_ranks[from_rank]
            msg = xa.Message(AID_STEPPER, "move_agents", agent_ids, dst_ranks)
            xa.send(rank, msg, flush=False)
        
        for rank in range(xa.WORLD_SIZE):
            msg = xa.Message(AID_STEPPER, "move_agents_done")
            xa.send(rank, msg, flush=False)

        xa.flush()

    def do_create_agents(self, new_agent_rank):
        """Send the create agent commands to steppers and update local registry."""
        for agent_id, constructor in self.agent_constructor:
            from_rank = self.agent_rank[agent_id]
            to_rank = new_agent_rank[agent_id]

            self.rank_agents[from_rank].remove(agent_id)
            self.rank_agents[to_rank].add(agent_id)
            self.agent_rank[agent_id] = to_rank

            msg = xa.Message(AID_STEPPER, "create_agent", agent_id, constructor)
            rank = new_agent_rank[agent_id]
            xa.send(rank, msg, flush=False)

        self.agent_constructor.clear()

        for rank in range(xa.WORLD_SIZE):
            msg = xa.Message(AID_STEPPER, "create_agent_done")
            xa.send(rank, msg, flush=False)

        xa.flush()

    def distribute(self):
        """Distribute agents across ranks."""
        self.log.info("Distributing agent.")

        agent_load = self.get_agent_load()
        rank_load = self.get_rank_load(agent_load)

        # Make a local copy
        new_rank_agents = [set(aids) for aids in self.rank_agents]
        new_agent_rank = dict(self.agent_rank)

        # Set of agents that are moving
        moving_agents = set()

        # Keep moving while imbalance is greater than tolerance
        # Or can't make any movements
        while True:
            imbalance = self.get_imbalance(rank_load)
            if imbalance < IMBALANCE_TOLERANCE:
                break

            self.log.info("Load imbalance %f", imbalance)

            agent_ids, rank_load, new_rank_agents, new_agent_rank = self.greedy_move(
                agent_load, rank_load, new_rank_agents, new_agent_rank
            )
            if not agent_ids:
                break

            moving_agents.update(agent_ids)

        self.log.info("%d agents to be created", len(self.agent_constructor))
        self.log.info("%d agents to be moved", len(self.agent_constructor))

        self.do_move_agents(moving_agents, new_agent_rank)
        self.do_create_agents(new_agent_rank)
