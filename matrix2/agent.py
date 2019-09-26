"""Agent implementations."""

from abc import ABC, abstractmethod


class Agent(ABC):
    """Interface of agent implementations."""

    def __init__(self, agent_id):
        """Initialize.

        Parameters
        ----------
            agent_id: string
                ID for the agent.
        """
        self.agent_id = agent_id

    @abstractmethod
    def step(self, timestep, timeperiod, incoming_messages):
        """Execute a timestep.

        Parameters
        ----------
            timestep: float
                The current timestep.
            timeperiod: a 2 tuple of floats (start, end)
                The realtime period corresponding to the given timestep.
            incoming_messages: list of 2 tuples [(src_id, message)]
                src_id: ID of the sender agent
                message: the actual message body.

        Returns
        -------
            outgoing_messages: list of 2 tuples [(dst_id, message)]
                dst_id: ID of the destination agent.
                message: the actual message body.
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
