"""Shared State Agent Implementations."""

from abc import ABC, abstractmethod

class SharedStateAgent(ABC):
    """Interface of shared state agent implementations."""

    def __init__(self, agent_id):
        """Initialize.

        Parameters
        ----------
            agent_id: string
                ID for the agent.
        """
        self.agent_id = agent_id

    @abstractmethod
    def step(self, timestep, timeperiod):
        """Execute a timestep.

        Parameters
        ----------
            timestep: float
                The current timestep.
            timeperiod: a 2 tuple of floats (start, end)
                The realtime period corresponding to the given timestep.

        Returns
        -------
            updates: list of three tuples [(store_name, order_key, update)]
                store_name: string
                    Name of the state store.
                order_key: string
                    Key used to order updates before application.
                update: store implementation specific
                    Data uses to update the store.
        """

    def memory_usage(self):
        """Return the memory usage of the agent.

        This method is called at the end of every step.

        Returns
        -------
            float
                Memory usage of the agent in bytes
        """
        return 1.0

    def is_alive(self):
        """Return if the agent is still alive.

        This method is called at the end of every step.
        It is the agent's responsibility to inform its neighbors that it is dead.
        Once this method returns false, the agent is deleted and garbage collected.

        Returns
        -------
            bool
                If false, the agent is considered dead.
        """
        return True
