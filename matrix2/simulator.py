"""Simulator implementations."""

from time import time
from abc import ABC, abstractmethod
from collections import defaultdict


class Simulator(ABC):
    """Simulator implementations."""

    @abstractmethod
    def __init__(
        self, agent_class, agent_population, timestep_generator, agent_distributor_class
    ):
        """Initialize."""

    @abstractmethod
    def run(self):
        """Run the simulation."""


class SingleProcessSimulator(Simulator):
    """Single process simulator."""

    def __init__(
        self, agent_class, agent_population, timestep_generator, agent_distributor_class
    ):
        """Initialize."""
        super().__init__(
            agent_class, agent_population, timestep_generator, agent_distributor_class
        )

        self.agent_class = agent_class
        self.agent_population = agent_population
        self.timestep_generator = timestep_generator

        self.agent_distributor = agent_distributor_class(
            self.agent_population, 1, [0, []]
        )

    def run(self):
        """Run the simulation."""
        agents = {}
        incoming_messages = defaultdict(list)

        while True:
            rank_step_start_time = time()

            # Get the current timestep
            timestep, timeperiod = self.timestep_generator.get_next_timestep()
            if timestep is None:
                return

            # Distribute any new agents
            for _rank, agent_tuples in self.agent_distributor.distribute(
                timestep, timeperiod
            ):
                for agent_id, in_neighbors, out_neighbors in agent_tuples:
                    agents[agent_id] = self.agent_class(
                        agent_id, in_neighbors, out_neighbors
                    )

            # Step through all the agents
            outgoing_messages = defaultdict(list)
            dead_agents = []
            for agent_id, agent in agents.items():
                agent_step_start_time = time()
                received_msgs = incoming_messages[agent_id]

                sent_msgs = agent.step(
                    timestep, timeperiod, incoming_messages[agent_id]
                )
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
            incoming_messages = outgoing_messages

            # Log the rank/process's performance
            rank_step_time = time() - rank_step_start_time
            self.agent_distributor.rank_step_profile(0, timestep, rank_step_time)
