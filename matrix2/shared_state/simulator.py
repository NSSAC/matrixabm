"""Shared state simulator using MPI Actors."""

import logging
from collections import defaultdict
from abc import ABC, abstractmethod
from dataclasses import dataclass

import xactor.mpi_actor as xa

from . import AID_POPULATION, AID_STEPPER
from .agent_stepper import AgentStepper
from .greedy_agent_coordinator import GreedyAgentCoordinator

log = logging.getLogger("%s.%d" % (__name__, xa.WORLD_RANK))


@dataclass(init=False)
class Constructor:
    """A object constructor."""

    cls: type  # The class to be used for creating the object
    args: list  # The positional arguments of the constructor
    kwargs: dict  # The keyword arguements of the constructor

    def __init__(self, cls, *args, **kwargs):
        """Make the consturctor."""
        self.cls = cls
        self.args = args
        self.kwargs = kwargs

    def construct(self):
        """Construct the object."""
        return self.cls(*self.args, **self.kwargs)


class Simulation(ABC):
    """Run a simulation."""

    def __init__(self):
        # Create the timestep generator
        constructor = self.get_timestep_generator()
        self.timestep_generator = constructor.construct()

        # Create the population
        constructor = self.get_population()
        msg = xa.Message(
            xa.RANK_AID,
            "create_actor",
            AID_POPULATION,
            constructor.cls,
            constructor.args,
            constructor.kwargs,
        )

        # Create the state stores
        self.store_ranks = defaultdict(list)
        for i, (store_name, constructor) in enumerate(self.get_state_stores()):
            for node in xa.get_nodes():
                ranks = xa.get_node_ranks(node)
                rank = ranks[i % len(ranks)]

                self.store_ranks[store_name].append(rank)

                msg = xa.Message(
                    xa.RANK_AID,
                    "create_actor",
                    store_name,
                    constructor.cls,
                    constructor.args,
                    constructor.kwargs,
                )
                xa.send(rank, msg)
        self.store_ranks = dict(self.store_ranks)

        # Create the steppers
        for rank in range(xa.WORLD_SIZE):
            msg = xa.Message(
                xa.RANK_AID,
                "create_actor",
                AID_STEPPER,
                AgentStepper,
                (self.store_ranks,),
            )

        self.timestep = None
        self.num_stores = sum(len(ranks) for ranks in self.store_ranks.values())
        self.num_store_flush_done = 0

    @abstractmethod
    def get_state_stores(self):
        """Return the state store constuctors.

        Returns
        -------
            A list of 2 tuples [(store_name, constructor)]
                store_name: str
                    name of the state store
                constructor: Constructor
                    constructor of the state store class
        """

    @abstractmethod
    def get_population(self):
        """Return the population constructor.

        Returns
        -------
            constructor: Constructor
                constructor of the agent population class
        """

    @abstractmethod
    def get_timestep_generator(self):
        """Return the timestep generator constructor.

        Returns
        -------
            constructor: Constructor
                constructor of the timestep generator class
        """

    def get_agent_coordinator(self):
        """Return the agent coordinator constructor.

        Returns
        -------
            constructor: Constructor
                constructor of the agent coordinator
        """
        return Constructor(GreedyAgentCoordinator, xa.WORLD_SIZE)

    def step(self):
        """Start the step."""
        self.timestep = self.timestep_generator.get_next_timestep()
        if self.timestep is None:
            xa.stop()
            return

        self.num_store_flush_done = 0

        msg = xa.Message(AID_POPULATION, "create_agents", self.timestep)
        xa.send(xa.WORLD_RANK, msg)

        msg = xa.Message(AID_STEPPER, "step", self.timestep)
        for rank in range(xa.WORLD_SIZE):
            xa.send(rank, msg)

    def store_flush_done(self):
        """When all stores are done start next step."""
        self.num_store_flush_done += 1
        if self.num_store_flush_done == self.num_stores:
            self.step()
