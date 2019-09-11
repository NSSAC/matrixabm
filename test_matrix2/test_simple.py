"""Simple simulation."""

from matrix2.agent import Agent
from matrix2.agent_population import FixedPopulation
from matrix2.timestep_generator import RangeTimestepGenerator
from matrix2.agent_distributor import RoundRobinAgentDistributor
from matrix2.simulator import SingleProcessSimulator


class SimpleAgent(Agent):
    """Simple agent."""

    def __init__(self, *args, **kwargs):
        """Initialize."""

        self.n_agents = kwargs.pop("n_agents")
        self.n_timesteps = kwargs.pop("n_timesteps")
        self._is_alive = True

        super().__init__(*args, **kwargs)

    def step(self, timestep, _timeperiod, incoming_messages):
        """Sleep between 1 second."""

        if timestep == 0:
            assert len(incoming_messages) == 0
        else:
            assert len(incoming_messages) == self.n_agents

        if timestep == float(self.n_timesteps - 1):
            self._is_alive = False

        return [(i, "hello") for i in range(self.n_agents)]

    def is_alive(self):
        return self._is_alive


def test_simple_simulation():
    """Run an instance of the simple simulation."""
    n_agents = 10
    n_timesteps = 10

    simulator = SingleProcessSimulator(
        agent_class=SimpleAgent,
        agent_kwargs={"n_agents": n_agents, "n_timesteps": n_timesteps},

        agent_population_class=FixedPopulation,
        agent_population_kwargs={"agent_ids": n_agents},

        timestep_generator_class=RangeTimestepGenerator,
        timestep_generator_kwargs={"n_timesteps": n_timesteps},

        agent_distributor_class=RoundRobinAgentDistributor,
    )

    simulator.run()
