#!/usr/bin/env python3
"""The BluePill Simulation.

In the BluePill simulation
agents choose change their state in sequence:
rock -> paper -> scissors -> rock.
The starting state of every agent is chosen at random.
State of all agents is stored in a single SQLite3 state store.
Agent objects themselves store not state information.
"""

import random
import logging

import click

from matrixabm import (
    asys,
    WORLD_SIZE,
    Agent,
    AgentPopulation,
    RangeTimestepGenerator,
    GreedyLoadBalancer,
    StateUpdate,
    SQLite3Store,
    SQLite3Manager,
    Coordinator,
    Runner,
    Simulator,
    TensorboardWriter,
    Constructor,
)


# Actor IDs
AID_MAIN = "main"
AID_SIMULATOR = "simulator"
AID_COORDINATOR = "coordinator"
AID_RUNNER = "runner"
AID_POPULATION = "population"
AID_TIMESTEP_GEN = "timestep_gen"
AID_SUMMARY_WRITER = "summary_writer"
AID_SQLITE3 = "sqlite3"

# The database schema name
STORE_NAME = "bluepill"


class BluePillStore(SQLite3Store):
    """The BluePill Store.

    The BluePill store maintains the state of the simulation.
    The store object is a SQLite3 file.
    The file contains one table called "state".
    """

    def __init__(self):
        """Initialize."""
        super().__init__(STORE_NAME, AID_SIMULATOR, AID_SQLITE3)

        # Setup the state table
        con = asys.local_actor(AID_SQLITE3).connection
        sql = f"""
            create table if not exists
            {self.store_name}.state (
                agent_id text,
                state text,
                timestep float
            )"""
        con.execute(sql)

    def set_state(self, agent_id, state, step):
        """Set the agent state.

        Parameters
        ----------
        agent_id : str
            ID of the agent
        state : str
            State of the agent
            One of "rock", "paper", and "scissors"
        step : int
            The current timestep
        """
        con = asys.local_actor(AID_SQLITE3).connection
        sql = f"insert into {self.store_name}.state values (?,?,?)"
        con.execute(sql, (agent_id, state, step))

    @staticmethod
    def get_state(store_name, agent_id):
        """Get the latest state of the agent.

        Parameters
        ----------
        store_name : str
            Name of the SQLite3 database
        agent_id : str
            ID of the agent

        Returns
        -------
        str or None
            The latest state of the given agent.
        """
        con = asys.local_actor(AID_SQLITE3).connection
        sql = f"""
            select state
            from {store_name}.state
            where agent_id = ?
            order by timestep desc
            limit 1
            """
        row = con.execute(sql, (agent_id,)).fetchone()
        if row is None:
            return None

        return row[0]


class BluePillAgent(Agent):
    """The BluePill Agent.

    The BluePill agent has the following behavior.
    At start its state is chosen at random from ["rock", "paper", and "scissors"]
    Every following timestep its state cycles in the given order:
    rock -> paper -> scissors -> rock.
    """

    def __init__(self, store_name, agent_id):
        self.agent_id = agent_id
        self.store_name = store_name
        self.state = random.choice(["rock", "paper", "scissors"])

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


class BluePillPopulation(AgentPopulation):
    """Blue Pill Agent Population.

    At every timestep between 100 and 200 new agents are created.
    """

    def __init__(self):
        super().__init__(AID_COORDINATOR)

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


class Main:
    """The main actor."""

    def __init__(self, store_path, summary_dir):
        """Initialize.

        store_path : str
            Path to the local sqlite3 file
        summary_dir : str
            Directory where runtime summary is to be written
        """
        self.store_path = store_path
        self.summary_dir = summary_dir

    def main(self):
        """Setup the connectors on the ranks."""
        # Create the simulator
        asys.create_actor(
            asys.MASTER_RANK,
            AID_SIMULATOR,
            Simulator,
            coordinator_aid=AID_COORDINATOR,
            runner_aid=AID_RUNNER,
            population_aid=AID_POPULATION,
            timestep_generator_aid=AID_TIMESTEP_GEN,
            store_names=[STORE_NAME],
        )

        # Create the coordinator
        load_balancer = GreedyLoadBalancer(WORLD_SIZE)
        asys.create_actor(
            asys.MASTER_RANK,
            AID_COORDINATOR,
            Coordinator,
            load_balancer,
            AID_SIMULATOR,
            AID_RUNNER,
            AID_SUMMARY_WRITER,
        )

        # Create the SQLite3 connection managers on every rank
        for rank in asys.ranks():
            asys.create_actor(
                rank, AID_SQLITE3, SQLite3Manager, [STORE_NAME], [self.store_path]
            )

        # Create a store on the first rank of every node
        store_ranks = {STORE_NAME: []}
        for node in asys.nodes():
            rank = asys.node_ranks(node)[0]
            store_ranks[STORE_NAME].append(rank)
        store_proxies = {
            name: asys.ActorProxy(store_ranks[name], STORE_NAME)
            for name, ranks in store_ranks.items()
        }
        store_proxies[STORE_NAME].create_actor_(BluePillStore)

        # Create the runners on every rank
        for rank in asys.ranks():
            asys.create_actor(
                rank, AID_RUNNER, Runner, store_proxies, AID_COORDINATOR, AID_RUNNER
            )

        # Create the timestep generator
        asys.create_actor(
            asys.MASTER_RANK, AID_TIMESTEP_GEN, RangeTimestepGenerator, 10
        )

        # Create the agent population
        asys.create_actor(asys.MASTER_RANK, AID_POPULATION, BluePillPopulation)

        # Create the tensorboard summary directory manager
        asys.create_actor(
            asys.MASTER_RANK, AID_SUMMARY_WRITER, TensorboardWriter, self.summary_dir
        )

        # Start the simulator
        asys.ActorProxy(asys.MASTER_RANK, AID_SIMULATOR).start()


@click.command()
@click.argument("store_path")
@click.argument("summary_dir")
def main(store_path, summary_dir):
    """Run a Blue Pill simulation.

    STORE_PATH should point the file when the SQLite3 store file should be created.
    SUMMARY_DIR should point to the directory where runtime summary is to be written.
    """
    asys.start(AID_MAIN, Main, store_path, summary_dir)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()  # pylint: disable=no-value-for-parameter
