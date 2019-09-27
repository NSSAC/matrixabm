"""Shared state simulator implementations."""
# pylint: disable=dangerous-default-value

from time import time
import logging
from abc import ABC, abstractmethod

log = logging.getLogger(__name__)

class SharedStateSimulator(ABC):
    """Shared state simulator implementation interface."""

    @abstractmethod
    def __init__(
        self,
        agent_class=None,
        agent_population_class=None,
        agent_population_kwargs={},
        timestep_generator_class=None,
        timestep_generator_kwargs={},
        agent_distributor_class=None,
        agent_distributor_kwargs={},
        state_store_classes=[],
        state_store_kwargs=[]
    ):
        """Initialize."""
        self.agent_class = agent_class
        self.agent_population_class = agent_population_class
        self.agent_population_kwargs = agent_population_kwargs
        self.timestep_generator_class = timestep_generator_class
        self.timestep_generator_kwargs = timestep_generator_kwargs
        self.agent_distributor_class = agent_distributor_class
        self.agent_distributor_kwargs = agent_distributor_kwargs
        self.state_store_classes = state_store_classes
        self.state_store_kwargs = state_store_kwargs

        if self.agent_class is None:
            raise ValueError("Agent class not specified")
        if self.agent_population_class is None:
            raise ValueError("Agent population class not specified")
        if self.timestep_generator_class is None:
            raise ValueError("Timestep generator class not specified")
        if self.agent_distributor_class is None:
            raise ValueError("Agent distributor class not specified")
        if len(self.state_store_classes) != len(self.state_store_kwargs):
            raise ValueError("Mismatch between the state store classes and their constructor kwargs")

    @abstractmethod
    def run(self):
        """Run the simulation."""

class SingleProcessSharedStateSimulator(SharedStateSimulator):
    """Single process shared state simulator."""

    def __init__(self, *args, **kwargs):
        """Initialize."""
        super().__init__(*args, **kwargs)

        self.agent_population = self.agent_population_class(
            **self.agent_population_kwargs
        )

        self.timestep_generator = self.timestep_generator_class(
            **self.timestep_generator_kwargs
        )

        self.agent_distributor = self.agent_distributor_class(
            self.agent_population, 1, [0, []], **self.agent_distributor_kwargs
        )

        self.state_stores = {}
        for class_, kwargs in zip(self.state_store_classes, self.state_store_kwargs):
            store = class_(**kwargs)
            self.state_stores[store.store_name] = store

    def run(self):
        """Run the simulation."""
        sim_start_time = time()

        agents = {}

        # Run the init function
        for store in self.state_stores.values():
            store.run_init()

        while True:
            rank_step_start_time = time()

            # Get the current timestep
            timestep, timeperiod = self.timestep_generator.get_next_timestep()
            if timestep is None:
                sim_end_time = time()
                log.info("Total sim runtime: %f seconds", sim_end_time - sim_start_time)
                return

            # Log the start of timestep
            log.info("Starting timestep %f: (%f, %f)", timestep, timeperiod[0], timeperiod[1])

            # Distribute any new agents
            for _rank, agent_constructor_args_list in self.agent_distributor.distribute(
                timestep, timeperiod
            ):
                for agent_id, agent_kwargs in agent_constructor_args_list:
                    agents[agent_id] = self.agent_class(agent_id, **agent_kwargs)

            # Step through all the agents
            dead_agents = []
            for agent_id, agent in agents.items():
                agent_step_start_time = time()
                updates = agent.step(timestep, timeperiod)
                for store_name, order_key, update in updates:
                    self.state_stores[store_name].handle_update(order_key, update)

                if not agent.is_alive():
                    dead_agents.append(agent_id)

                # Log the agent's performance
                agent_step_time = time() - agent_step_start_time
                memory_usage = agent.memory_usage()
                num_updates = len(updates)
                self.agent_distributor.agent_step_profile(
                    agent_id,
                    timestep,
                    agent_step_time,
                    memory_usage,
                    None,
                    None,
                    num_updates
                )

            # Flush the stores
            for store in self.state_stores.values():
                store.flush()

            # Run the maintainance function
            for store in self.state_stores.values():
                store.run_maint()

            # Remove any dead agents
            for agent_id in dead_agents:
                self.agent_distributor.agent_died(agent_id, timestep, timeperiod)
                del agents[agent_id]

            # Log the rank/process's performance
            rank_step_time = time() - rank_step_start_time
            self.agent_distributor.rank_step_profile(0, timestep, rank_step_time)

