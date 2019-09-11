"""Simulator implementations."""
# pylint: disable=dangerous-default-value

from time import time
from abc import ABC, abstractmethod
from collections import defaultdict


class Simulator(ABC):
    """Simulator implementations."""

    @abstractmethod
    def __init__(
        self,
        agent_class=None,
        agent_kwargs={},
        agent_population_class=None,
        agent_population_kwargs={},
        timestep_generator_class=None,
        timestep_generator_kwargs={},
        agent_distributor_class=None,
        agent_distributor_kwargs={},
    ):
        """Initialize."""
        self.agent_class = agent_class
        self.agent_kwargs = agent_kwargs
        self.agent_population_class = agent_population_class
        self.agent_population_kwargs = agent_population_kwargs
        self.timestep_generator_class = timestep_generator_class
        self.timestep_generator_kwargs = timestep_generator_kwargs
        self.agent_distributor_class = agent_distributor_class
        self.agent_distributor_kwargs = agent_distributor_kwargs

        if self.agent_class is None:
            raise ValueError("Agent class not specified")
        if self.agent_population_class is None:
            raise ValueError("Agent population class not specified")
        if self.timestep_generator_class is None:
            raise ValueError("Timestep generator class not specified")
        if self.agent_distributor_class is None:
            raise ValueError("Agent distributor class not specified")

    @abstractmethod
    def run(self):
        """Run the simulation."""


class SingleProcessSimulator(Simulator):
    """Single process simulator."""

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

    def run(self):
        """Run the simulation."""
        agents = {}
        all_received_messages = defaultdict(list)

        while True:
            rank_step_start_time = time()

            # Get the current timestep
            timestep, timeperiod = self.timestep_generator.get_next_timestep()
            if timestep is None:
                return

            # Distribute any new agents
            for _rank, agent_constructor_args_list in self.agent_distributor.distribute(
                timestep, timeperiod
            ):
                for agent_id, agent_kwargs in agent_constructor_args_list:
                    kwargs = dict(self.agent_kwargs)
                    kwargs.update(agent_kwargs)
                    agents[agent_id] = self.agent_class(agent_id, **kwargs)

            # Step through all the agents
            outgoing_messages = defaultdict(list)
            dead_agents = []
            for agent_id, agent in agents.items():
                agent_step_start_time = time()
                received_msgs = all_received_messages[agent_id]

                sent_msgs = agent.step(timestep, timeperiod, received_msgs)
                for dst_id, message in sent_msgs:
                    outgoing_messages[dst_id].append((agent_id, message))

                if not agent.is_alive():
                    dead_agents.append(agent_id)

                # Log the agent's performance
                agent_step_time = time() - agent_step_start_time
                memory_usage = agent.memory_usage()
                received_msg_sizes = [
                    (src_id, len(message)) for src_id, message in received_msgs
                ]
                sent_msg_sizes = [
                    (dst_id, len(message)) for dst_id, message in sent_msgs
                ]
                self.agent_distributor.agent_step_profile(
                    agent_id,
                    timestep,
                    agent_step_time,
                    memory_usage,
                    received_msg_sizes,
                    sent_msg_sizes,
                )

            # Remove any dead agents
            for agent_id in dead_agents:
                self.agent_distributor.agent_died(agent_id, timestep, timeperiod)
                del agents[agent_id]

            # Outgoint messages of the current step are incoming for the next
            all_received_messages = outgoing_messages

            # Log the rank/process's performance
            rank_step_time = time() - rank_step_start_time
            self.agent_distributor.rank_step_profile(0, timestep, rank_step_time)
