"""Agent distributor implementations."""

from abc import ABC, abstractmethod
from itertools import cycle
from collections import defaultdict


class AgentDistributor(ABC):
    """Interface of agent distributor implementations."""

    def __init__(self, population, n_ranks, rank_neighbors):
        """Initialize.

        Parameters
        ----------
            population: .agent_population.Population
                Agent population
            n_ranks: int
                Number of ranks
            rank_neighbors: list of 2 tuples [(rank_u, rank_vs)]
                Undirected weighted edgelist representing the rank network

                rank_u: int
                    Rank of the source vertex
                rank_vs: list of 2 tuples [(rank_v, weight)]
                    rank_v: int
                        rank of the destination vertex
                    weight: float
                        Connection weight
        """
        self.population = population
        self.n_ranks = n_ranks
        self.rank_neighbors = rank_neighbors

    def agent_died(self, agent_id, timestep, timeperiod):
        """Mark the agent as dead.

        Parameters
        ----------
            agent_id: string
                ID of the dead agent.
            timestep: float
                The current (just finished) timestep.
            timeperiod: a tuple of floats (start, end)
                The realtime period corresponding to the given timestep.
        """
        self.population.agent_died(agent_id, timestep, timeperiod)

    @abstractmethod
    def distribute(self, timestep, timeperiod):
        """Distribute new agent ids across processors and ranks.

        Returns
        -------
            rank_agent_tuples: list of 2 tuples [(rank, agent_tuples)]
                rank: rank of the compute process.
                agent_tuples: list of three tuples [(agent_id, in_neighbors, out_neighbors)]
                    agent_id: string
                        ID for the agent.
                    in_neighbors: list of two tuples [(neighbor_id, weight)] or None
                        neighbor_id: string
                            ID of incoming neighbor
                        weight: float
                            Connection weight with the neighbor
                    out_neighbors: list of two tuples [(neighbor_id, weight)] or None
                        neighbor_id: string
                            ID of outgoing neighbor
                        weight: float
                            Connection weight with the neighbor
        """

    def agent_step_profile(
        self,
        agent_id,
        timestep,
        step_time,
        memory_usage,
        received_msg_sizes,
        sent_msg_sizes,
    ):
        """Note the compute and message load of an agent on a given timestep.

        Parameters
        ----------
            agent_id: string
                ID of the given agent.
            timestep: float
                The current timestep.
            step_time: float
                Compute time takes to execute the step (in seconds).
            memory_usage: float
                Size of agent's memory
            received_msg_sizes: list of 2 tuples [(src_id, msg_size)]
                src_id: ID of the sender agent.
                msg_size: Size of the message.
            sent_msg_sizes: list of 2 tuples [(dst_id, msg_size)]
                dst_id: ID of the receiver agent.
                msg_size: Size of the message.
        """

    def rank_step_profile(self, rank, timestep, step_time):
        """Note the compute load of the rank on a given timestep.

        Parameters
        ----------
            rank: int
                rank of the compute process
            timestep: float
                The current timestep.
            step_time: float
                Compute time takes to execute the step (in seconds).
        """

    def is_load_imbalanced(self):
        """Return true if there is an imbalance of load across ranks.

        Returns
        -------
            bool
                If true, redistribution of agents is triggered.
        """
        return False

    def redistribute(self):
        """Redistriubte agent ids acrross processors and ranks.

        Returns
        -------
            agent_movement: list of 3 tuples [(src_rank, dst_rank, agent_ids)]
        """
        return []


class UniformAgentDistributor(AgentDistributor):
    """Distribute agent ids across machines uniformly."""

    def distribute(self, timestep, timeperiod):
        """Distribute new agent ids across processors and ranks."""
        new_agent_tuples = self.population.get_new_agents(timestep, timeperiod)
        if new_agent_tuples:
            rank_agent_tuples = defaultdict(list)
            for rank, agent_tuple in zip(cycle(range(self.n_ranks)), new_agent_tuples):
                rank_agent_tuples[rank].append(agent_tuple)
            return list(rank_agent_tuples.items())

        return []
