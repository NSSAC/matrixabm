"""Agent interface."""

from abc import ABC, abstractmethod

class Agent(ABC):
    """Agent type interface."""

    @abstractmethod
    def step(self, timestep):
        """Run a step of the agent code.

        Parameters
        ----------
            timestep: Timestep
                The current timestep.

        Returns
        -------
            List of store updates
        """

    @abstractmethod
    def is_alive(self):
        """Return True if the agent is still alive, False otherwise."""

    @abstractmethod
    def memory_usage(self):
        """Return the memory usage of the agent in bytes."""
