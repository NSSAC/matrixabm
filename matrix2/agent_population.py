"""Agent Population implementations."""

from abc import ABC, abstractmethod


class Population(ABC):
    """Agent population interface."""

    @abstractmethod
    def get_new_agents(self, timestep, timeperiod):
        """Return the list of agents joining in the current timestep.

        Parameters
        ----------
            timestep: float
                The current (about to start) timestep.
            timeperiod: a tuple of floats (start, end)
                The realtime period corresponding to the given timestep.

        Returns
        -------
            list of three tuples [(agent_id, in_neighbors, out_neighbors)]
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


class FixedDisconnectedPopulation(Population):
    """Fixed and disconnected agent population."""

    def __init__(self, agent_ids):
        """Initialize.

        Parameters
        ----------
            agent_ids: iterable or int
                If agent_ids is of type int range(agent_ids) is used
        """
        if isinstance(agent_ids, int):
            agent_ids = range(agent_ids)

        self.agent_ids = list(agent_ids)

    def get_new_agents(self, timestep, _timeperiod):
        """Return all agents at timestep 0."""
        if timestep == 0:
            return [(agent_id, None, None) for agent_id in self.agent_ids]

        return []
