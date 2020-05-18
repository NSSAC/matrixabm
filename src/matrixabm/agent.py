"""Agent and Agent Population interface."""

from abc import ABC, abstractmethod

from . import asys


class Agent(ABC):
    """Agent interface.

    The Agent interface models a single agent in the simulation.
    Agents in the Matrix are not actors themselves.
    They are managed by a agent runner actor.
    The agent runner actor calls the `step`, `is_alive` and `memory_usage`
    methods of the agent at each timestep.
    """

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
        """Return the memory usage of the agent.

        Returns
        -------
        float
            The memory usage of the agent.
            This can be a relative number.
        """


class AgentPopulation(ABC):
    """Agent population interface.

    The agent population models births of agents.
    There is a single agent population actor in every simulation.
    At the beginning of every timestep
    the simulator actor sends it the `create_agents` message
    On receiving the above message the population actor
    is supposed send a number of `create_agent` messages
    to the coordinator actor,
    one for every agent created.
    Once all the `create_agent` messages are sent,
    the population actor sends the `create_agent_done` message
    to the coordinator actor.

    Receives
    --------
    * `create_agents` from Simulator

    Sends
    -----
    * `create_agent*` to Coordinator
    * `create_agent_done` to Coordinator
    """

    def __init__(self, coordinator_aid):
        """Initialize.

        Parameters
        ----------
        coordinator_aid : str
            ID of the agent coordinator actor
        """
        self.coordinator_proxy = asys.ActorProxy(asys.MASTER_RANK, coordinator_aid)

    def create_agents(self, timestep):
        """Create new agents in the simulation.

        Parameters
        ----------
        timestep : Timestep
            The current (to start) timestep
        """
        for args in self.do_create_agents(timestep):
            self.coordinator_proxy.create_agent(*args, buffer_=True)

        self.coordinator_proxy.create_agent_done()

    @abstractmethod
    def do_create_agents(self, timestep):
        """Create new agents in the simulation.

        Parameters
        ----------
        timestep : Timestep
            The current (about to start) timestep.

        Returns
        -------
        iterable of 4 tuples [(agent_id, constructor, step_time, memory_usage)]
            agent_id : str
                ID of the to be created agent
            constructor : Constructor
                Constructor of the agent
            step_time : float
                Initial estimate step_time per unit simulated real time (in seconds)
            memory_usage : float
                Initial estimate of memory usage
        """
