"""Agent interface.

The agent models a single agent in the simulation.

NOTE: Agents in the Matrix are not actors themselves.
They are contained in a agent runner actor.
The agent runner actor calls the `step', `is_alive' and `memory_usage'
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
            timestep: Timestep
                The current timestep

        Returns
        -------
            An iterable of StateUpdate objects
        """

    @abstractmethod
    def is_alive(self):
        """Return True if the agent is still alive, False otherwise."""

    @abstractmethod
    def memory_usage(self):
        """Return the memory usage of the agent in bytes."""
