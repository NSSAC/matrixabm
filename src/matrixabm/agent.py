"""Agent interface.

The Agent interface models single agents in the simulation.

NOTE: Agents in the Matrix are not actors themselves.
They are contained in a agent runner actor.
The agent runner actor calls
the `step`, `is_alive` and `memory_usage`
methods of the agent.
"""

from abc import ABC, abstractmethod

class Agent(ABC):
    """Agent interface."""

    @abstractmethod
    def step(self, timestep):
        """Run a step of the agent code.

        Parameters
        ----------
        timestep : Timestep
            The current timestep

        Returns
        -------
        updates : iterable of StateUpdate
            An iterable of state updates
        """

    @abstractmethod
    def is_alive(self):
        """Check if the agent is still "alive".

        Returns
        -------
        bool
            True if agent is still alive
            False otherwise
        """

    @abstractmethod
    def memory_usage(self):
        """Check the memory usage of the agent.

        Returns
        -------
        float
            The memory usage of the agent.
            This can be a relative number.
        """
