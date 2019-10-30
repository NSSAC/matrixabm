"""Agent population implementations."""

import logging
from abc import ABC, abstractmethod

import xactor.mpi_actor as xa


class Population(ABC):
    """Agent population interface."""

    def __init__(self):
        """Initialize."""
        self.log = logging.getLogger(
            "%s(%d)" % (self.__class__.__name__, xa.WORLD_RANK)
        )

    @abstractmethod
    def create_agents(self, timestep):
        """Create new agents in the simulation.

        This method sends "agent_created" messages to the COORDINATOR
        informing it of the new agents to be created in the current timestep.

        Finally it sends "distribute" message to the COORDINATOR.

        Parameters
        ----------
            timestep: Timestep
                The current (about to start) timestep.
        """

    def agent_died(self, agent_id):
        """Note the death of the agent.

        Parameters
        ----------
            agent_id: ID of the dead agent.
        """
        if __debug__:
            self.log.debug("Agent %d died", agent_id)
