#!/usr/bin/env python3
"""The BluePill Simulation."""

import random
import logging

import click

from matrixabm import asys, AID_MAIN
from matrixabm import Constructor, StateUpdate
from matrixabm import Agent, Population, Simulator
from matrixabm import RangeTimestepGenerator
# from matrixabm import RandomLoadBalancer
from matrixabm import GreedyLoadBalancer
from matrixabm import SQLite3Store, SQLite3Connector

STORE_NAME = "bluepill"
WORLD_SIZE = len(asys.ranks())


class BluePillStore(SQLite3Store):
    """The BluePill State Store."""

    def __init__(self, store_name, connector_name):
        """Initialize."""
        super().__init__(store_name, connector_name)

        self.setup()

    def setup(self):
        """Setup the state table."""
        con = asys.local_actor("connector").connection
        sql = f"""
            create table if not exists
            {self.store_name}.state (
                agent_id text,
                state text,
                timestep float
            )"""
        con.execute(sql)

    def set_state(self, agent_id, state, step):
        """Set the agent state."""
        con = asys.local_actor("connector").connection
        sql = f"insert into {self.store_name}.state values (?,?,?)"
        con.execute(sql, (agent_id, state, step))

    @staticmethod
    def get_state(store_name, agent_id):
        """Get the state of the agent."""
        con = asys.local_actor("connector").connection
        sql = f"""
            select state
            from {store_name}.state
            where agent_id = ?
            order by timestep desc
            limit 1
            """
        row = con.execute(sql, (agent_id,)).fetchone()
        if row is None:
            return random.choice(["rock", "paper", "scissors"])
        else:
            return row[0]


class BluePillAgent(Agent):
    """The BluePill Agent."""

    def __init__(self, store_name, agent_id):
        self.agent_id = agent_id
        self.store_name = store_name
        self.state = BluePillStore.get_state(self.store_name, self.agent_id)

    def step(self, timestep):
        """Return the step update."""
        # Compute local state change
        if self.state == "rock":
            self.state = "paper"
        elif self.state == "paper":
            self.state = "scissors"
        else:  # self.state == "scissors"
            self.state = "paper"

        # Compute order key of next state
        order_key = f"{timestep.step}-{self.agent_id}"

        # Make the update
        update = StateUpdate(
            self.store_name,
            order_key,
            "set_state",
            self.agent_id,
            self.state,
            timestep.step,
        )

        return [update]

    def memory_usage(self):
        """Return the memory usage."""
        return 1.0

    def is_alive(self):
        """Return if the agent is alive or not."""
        return random.choice((True, False))


class BluePillPopulation(Population):
    """Blue Pill Agent Population."""

    def do_create_agents(self, timestep):
        """Create the new agents for this timestep."""
        # Decide on the number of agents to create
        n = random.randint(100, 200)

        ret = []
        for i in range(n):
            agent_id = f"agent-{timestep.step}-{i}"
            constructor = Constructor(BluePillAgent, STORE_NAME, agent_id)
            step_time = 1.0
            memory_usage = 1.0
            ret.append((agent_id, constructor, step_time, memory_usage))

        return ret


class BluePillSimulator(Simulator):
    """Blue Pill Simulator."""

    def __init__(self, store_path):
        """Initialize."""
        super().__init__("tensorboard")

        self.store_path = store_path

    def TimestepGenerator(self):
        """Get the timestep generator constructor."""
        return Constructor(RangeTimestepGenerator, 10)

    def Population(self):
        """Get the population constructor."""
        return Constructor(BluePillPopulation)

    def StateStores(self):
        """Get the state store constructors."""
        ctor = Constructor(BluePillStore, STORE_NAME, "connector")
        return [(STORE_NAME, ctor)]

    def LoadBalancer(self):
        """Get the load balancer constructor."""
        # return Constructor(RandomLoadBalancer, WORLD_SIZE)
        return Constructor(GreedyLoadBalancer, WORLD_SIZE)

    def main(self):
        """Setup the connectors on the ranks."""
        for rank in asys.ranks():
            asys.create_actor(
                rank, "connector", SQLite3Connector, [STORE_NAME], [self.store_path]
            )
        asys.ActorProxy(asys.EVERY_RANK, "connector").connect()

        super().main()


@click.command()
@click.argument("store_path")
def main(store_path):
    """Run the simulation."""

    asys.start(AID_MAIN, BluePillSimulator, store_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()  # pylint: disable=no-value-for-parameter
