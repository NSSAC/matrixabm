"""Simple simulation."""

import time

from matrix2.agent import Agent
from matrix2.agent_population import FixedPopulation
from matrix2.timestep_generator import RangeTimestepGenerator
from matrix2.agent_distributor import RoundRobinAgentDistributor
from matrix2.simulator import SingleProcessSimulator
from matrix2.mpi_simulator import MPISimulator


class SimpleAgent(Agent):
    """Simple agent."""

    n_agents = None
    n_timesteps = None

    def __init__(self, *args, **kwargs):
        """Initialize."""
        self._is_alive = True

        super().__init__(*args, **kwargs)

    def step(self, timestep, _timeperiod, incoming_messages):
        """Say hello to everyone."""

        if timestep == 0:
            assert len(incoming_messages) == 0
        else:
            assert len(incoming_messages) == self.n_agents

        if timestep == float(self.n_timesteps - 1):
            self._is_alive = False

        time.sleep(0.001)

        return [(dst_id, (self.agent_id, timestep, "hello")) for dst_id in range(self.n_agents)]

    def is_alive(self):
        return self._is_alive


def test_single_process():
    """Run simple simulation on single process."""
    n_agents = 100
    n_timesteps = 100

    SimpleAgent.n_agents = n_agents
    SimpleAgent.n_timesteps = n_timesteps

    simulator = SingleProcessSimulator(
        agent_class=SimpleAgent,
        agent_population_class=FixedPopulation,
        agent_population_kwargs={"agent_ids": n_agents},
        timestep_generator_class=RangeTimestepGenerator,
        timestep_generator_kwargs={"n_timesteps": n_timesteps},
        agent_distributor_class=RoundRobinAgentDistributor,
    )

    simulator.run()


def test_mpi():
    """Run simple simulation on multiprocess."""
    n_agents = 100
    n_timesteps = 100

    SimpleAgent.n_agents = n_agents
    SimpleAgent.n_timesteps = n_timesteps

    simulator = MPISimulator(
        agent_class=SimpleAgent,
        agent_population_class=FixedPopulation,
        agent_population_kwargs={"agent_ids": n_agents},
        timestep_generator_class=RangeTimestepGenerator,
        timestep_generator_kwargs={"n_timesteps": n_timesteps},
        agent_distributor_class=RoundRobinAgentDistributor,
    )

    simulator.run()
