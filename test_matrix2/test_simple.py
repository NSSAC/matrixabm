"""Simple simulation."""

import time

from matrix2.agent import Agent
from matrix2.agent_population import FixedDisconnectedPopulation
from matrix2.timestep_generator import RangeTimestepGenerator
from matrix2.agent_distributor import UniformAgentDistributor
from matrix2.simulator import SingleProcessSimulator


class SimpleAgent(Agent):
    """Simple agent."""

    def __init__(self, *args, **kwargs):
        """Initialize."""
        super().__init__(*args, **kwargs)

        self._is_alive = True

    def step(self, timestep, _timeperiod, _incoming_messages):
        """Sleep between 1 second."""
        time.sleep(1.0)

        if timestep == 1.0:
            self._is_alive = False

        return []

    def is_alive(self):
        return self._is_alive


def test_simple_simulation():
    """Run an instance of the simple simulation."""
    n_agents = 2
    n_timesteps = 2

    simulator = SingleProcessSimulator(
        agent_class=SimpleAgent,

        agent_population_class=FixedDisconnectedPopulation,
        agent_population_kwargs={"agent_ids": n_agents},

        timestep_generator_class=RangeTimestepGenerator,
        timestep_generator_kwargs={"n_timesteps": n_timesteps},

        agent_distributor_class=UniformAgentDistributor,
    )

    simulator.run()
