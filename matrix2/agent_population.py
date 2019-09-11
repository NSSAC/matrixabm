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
            list of two tuples [(agent_id, agent_kwargs)]
                agent_id: string
                    ID for the agent.
                agent_kwargs: dict
                    Extra keyword arguments to pass to the agent constructor.
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


class FixedPopulation(Population):
    """Fixed agent population."""

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
            return [(agent_id, {}) for agent_id in self.agent_ids]

        return []
