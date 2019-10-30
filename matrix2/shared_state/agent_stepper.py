"""Container for agents on a given process."""

import logging
from time import time

import xactor.mpi_actor as xa

from . import AID_STEPPER, AID_COORDINATOR, AID_POPULATION


class AgentStepper:
    """Container for agent objects."""

    def __init__(self, store_ranks):
        """Worker process."""
        self.local_agents = {}
        self.store_ranks = store_ranks

        self.log = logging.getLogger(
            "%s(%d)" % (self.__class__.__name__, xa.WORLD_RANK)
        )

        self.timestep = None
        self.flag_create_agent_done = False
        self.flag_move_agents_done = False
        self.num_receive_agent_done = 0

    def prepare_for_next_step(self):
        """Reset the variables."""
        self.timestep = None
        self.flag_create_agent_done = False
        self.flag_move_agents_done = False
        self.num_receive_agent_done = 0

    def can_start_step(self):
        """Check if the current step can be started."""
        if self.timestep is None:
            return False
        if not self.flag_create_agent_done:
            return False
        if not self.flag_move_agents_done:
            return False
        if self.num_receive_agent_done < xa.WORLD_SIZE:
            return False

        return True

    def create_agent(self, agent_id, constructor):
        """Create an agent locally."""
        if __debug__:
            if agent_id in self.local_agents:
                raise RuntimeError(
                    "Can't create agent; agent %s already exists" % agent_id
                )

        self.local_agents[agent_id] = constructor.construct()

    def create_agent_done(self):
        """Respond to create event done message from coordinator."""
        assert not self.flag_create_agent_done

        self.log.info("Received create agent done signal")

        self.flag_create_agent_done = True
        if self.can_start_step():
            self.do_step()

    def move_agents(self, agent_ids, dst_ranks):
        """Send local agents to destination ranks."""
        assert len(agent_ids) == len(dst_ranks)

        self.log.info("Moving out %d agents", len(agent_ids))

        for agent_id, dst_rank in zip(agent_ids, dst_ranks):
            if __debug__:
                if agent_id not in self.local_agents:
                    raise RuntimeError(
                        "Can't send agent; agent %s doesn't exist" % agent_id
                    )

            agent = self.local_agents[agent_id]
            msg = xa.Message(AID_STEPPER, "receive_agent", agent_id, agent)
            xa.send(dst_rank, msg, flush=False)

        for rank in range(xa.WORLD_SIZE):
            msg = xa.Message(AID_STEPPER, "receive_agent_done", xa.WORLD_RANK)
            xa.send(rank, msg, flush=False)

        xa.flush()

    def move_agents_done(self):
        """Respond to move agents done message from coordinator."""
        assert not self.flag_move_agents_done

        self.log.info("Received move agents done signal")

        self.flag_move_agents_done = True
        if self.can_start_step():
            self.do_step()

    def receive_agent(self, agent_id, agent):
        """Receive an agent from another worker process."""
        if __debug__:
            if agent_id in self.local_agents:
                raise RuntimeError(
                    "Can't receive agent; agent %s already exists" % agent_id
                )

        self.local_agents[agent_id] = agent

    def receive_agent_done(self, rank):
        """Respond to receive agent done message from other agent steppers."""
        assert self.num_receive_agent_done < xa.WORLD_SIZE

        self.log.info("Received receive_agent_done agents done signal from %d", rank)

        self.num_receive_agent_done += 1
        if self.can_start_step():
            self.do_step()

    def step(self, timestep):
        """Respond to the step signal from the simulator."""
        assert self.timestep is None

        self.log.info("Received timestep %s", timestep)

        self.timestep = timestep
        if self.can_start_step():
            self.do_step()

    def do_step(self):
        """Step through the local agents to produce updates."""
        self.log.info("Starting step")

        dead_agents = []

        # Step through the agents
        for agent_id, agent in self.local_agents.items():
            start_time = time()

            # Step through the agent
            updates = agent.step(self.timestep)
            memory_usage = agent.memory_usage()
            is_alive = agent.is_alive()
            if not is_alive:
                dead_agents.append(agent_id)

            # Send out the updates
            for update in updates:
                store_name = update.store_name
                msg = xa.Message(store_name, "handle_update", update)
                store_ranks = self.store_ranks[store_name]
                for rank in store_ranks:
                    xa.send(rank, msg, flush=False)
            end_time = time()

            # Inform the coordinator
            msg = xa.Message(
                actor_id=AID_COORDINATOR,
                method="agent_step_profile",
                agent_id=agent_id,
                step_time=(end_time - start_time),
                memory_usage=memory_usage,
                is_alive=is_alive,
                timestep=self.timestep,
            )
            xa.send(xa.MASTER_RANK, msg, flush=False)

            # Inform the population
            if not is_alive:
                msg = xa.Message(AID_POPULATION, "agent_died", xa.WORLD_RANK)
                xa.send(xa.MASTER_RANK, msg, flush=False)

        # Delete any dead agents
        for agent_id in dead_agents:
            del self.local_agents[agent_id]

        # Tell stores that we are done for this step
        for store_name, ranks in self.store_ranks.items():
            for rank in ranks:
                msg = xa.Message(store_name, "agent_stepper_done", xa.WORLD_RANK)
                xa.send(rank, msg, flush=False)

        # Flush out any pending messages
        xa.flush()

        self.prepare_for_next_step()
