"""Agent population interface.

The agent population models births of agents.
There is a single population actor in every simulation.
At the beginning of every timestep
the main actor sends it the `create_agents` message
On receiving the above message the population actor
is supposed send a number of `create_agent` messages
to the coordinator actor,
one for every agent created.
Once all the `create_agent` messages are sent,
the population actor sends the `create_agent_done` message
to the coordinator actor.
"""

from abc import ABC, abstractmethod

from .standard_actors import COORDINATOR

class Population(ABC):
    """Agent population.

    Receives
    --------
    * create_agents from Simulator

    Sends
    -----
    * create_agent* to Coordinator
    * create_agent_done to Coordinator
    """

    def create_agents(self, timestep):
        """Create new agents in the simulation.

        Sender
        ------
        * ``Simulator``

        Parameters
        ----------
        timestep : Timestep
            The current (to start) timestep
        """
        for args in self.do_create_agents(timestep):
            COORDINATOR.create_agent(*args, buffer_=True)

        COORDINATOR.create_agent_done()

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
