"""Simple shared state simulation."""

import time
import random

from matrix2.shared_state_agent import SharedStateAgent
from matrix2.shared_state_simulator import SingleProcessSharedStateSimulator
from matrix2.shared_state_store import SQLite3Store, get_sqlite3_connection

from matrix2.agent_population import FixedPopulation
from matrix2.timestep_generator import RangeTimestepGenerator
from matrix2.agent_distributor import RoundRobinAgentDistributor

def store_init(store):
    """Initialize the sqlite3 store."""
    con = get_sqlite3_connection()

    sql = f"create table {store.store_name}.event (agent_id integer, timestep float, state text)"
    con.execute(sql)

    sql = f"create table {store.store_name}.state_count (state text, count integer)"
    con.execute(sql)

def store_maint(store):
    """Run maintainance on the sqlite3 store."""
    con = get_sqlite3_connection()

    with con:
        sql = f"delete from {store.store_name}.state_count"
        con.execute(sql)

        sql = f"""
            insert into {store.store_name}.state_count
            select state, count(*) from {store.store_name}.event group by state
            """
        con.execute(sql)

class BluePillAgent(SharedStateAgent):
    """Simple shared state agent."""

    n_agents = None
    n_timesteps = None
    stores = []

    def __init__(self, *args, **kwargs):
        """Initialize."""
        super().__init__(*args, **kwargs)
        self.con = get_sqlite3_connection()

    def _get_current_state(self, store_name):
        """Get the current known state of the agent."""
        con = get_sqlite3_connection()
        sql = f"""
            select state
            from {store_name}.event
            where agent_id = ?
            order by timestep desc
            limit 1
        """
        cur = con.execute(sql, (self.agent_id,))
        row = cur.fetchone()
        if not row:
            return None
        return row[0]

    @staticmethod
    def _compute_next_state(current_state):
        """Compute next state given current state."""
        if current_state is None:
            return random.choice(["rock", "paper", "scissors"])

        next_state = {"rock": "paper", "paper": "scissors", "scissors": "rock"}
        next_state = next_state[current_state]
        return next_state

    def _make_update(self, timestep, store_name, current_state):
        """Compute updates for state store."""
        order_key = "%s-%f" % (self.agent_id, timestep)
        sql = f"insert into {store_name}.event values (?,?,?)"
        params = (self.agent_id, timestep, current_state)
        return (store_name, order_key, (sql, params))

    def step(self, timestep, _timeperiod):
        """Say hello to everyone."""
        print(self.agent_id, timestep, _timeperiod)
        time.sleep(0.001)

        updates = []
        for store_name in self.stores:
            current_state = self._get_current_state(store_name)
            next_state = self._compute_next_state(current_state)
            update = self._make_update(timestep, store_name, next_state)
            updates.append(update)

        return updates

    def is_alive(self):
        return True
        # return random.choice([True, False])

def test_single_process(tmp_path):
    """Run the simple shared state simulation on single process."""
    n_agents = 100
    n_timesteps = 100

    BluePillAgent.n_agents = n_agents
    BluePillAgent.n_timesteps = n_timesteps
    BluePillAgent.stores = ["store1", "store2"]

    store1_path = str(tmp_path / "bluepill_store1.sqlite3")
    store2_path = str(tmp_path / "bluepill_store2.sqlite3")

    Simulator = SingleProcessSharedStateSimulator

    simulator = Simulator(
        agent_class=BluePillAgent,
        agent_population_class=FixedPopulation,
        agent_population_kwargs=dict(agent_ids=n_agents),
        timestep_generator_class=RangeTimestepGenerator,
        timestep_generator_kwargs=dict(n_timesteps=n_timesteps),
        agent_distributor_class=RoundRobinAgentDistributor,
        state_store_classes=[SQLite3Store, SQLite3Store],
        state_store_kwargs=[
            dict(store_name=BluePillAgent.stores[0], dsn=store1_path, init_func=store_init, maint_func=store_maint),
            dict(store_name=BluePillAgent.stores[1], dsn=store2_path, init_func=store_init, maint_func=store_maint),
        ]
    )

    simulator.run()
