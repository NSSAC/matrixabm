"""MPI Simulatior implementation."""

from time import time
from enum import IntEnum, auto
import queue
import threading
from collections import defaultdict
import logging as log

from mpi4py import MPI

from .simulator import Simulator

COMM_WORLD = MPI.COMM_WORLD
HOSTNAME = MPI.Get_processor_name()
WORLD_RANK = COMM_WORLD.Get_rank()
WORLD_SIZE = COMM_WORLD.Get_size()
MASTER_RANK = 0


class Message(IntEnum):
    """Message types."""

    TERMINATE = auto()

    AGENT_TO_AGENT = auto()
    AGENT_DIED = auto()

    AGENT_STEP_PROFILE = auto()
    RANK_STEP_PROFILE = auto()

    WORKER_DONE = auto()


class Comm:
    """Communicator wrapper."""

    def __init__(self):
        """Initialize."""
        self.msgq = queue.Queue()
        self.pending_sends = []
        self.receiver_thread = threading.Thread(target=self._receiver_thread)

        self.receiver_thread.start()

    def _cleanup_pending_sends(self):
        """Cleanup any pending sends that are done."""
        if self.pending_sends:
            indices = MPI.Request.Waitsome(self.pending_sends)
            if indices is not None:
                self.pending_sends = [
                    r for i, r in enumerate(self.pending_sends) if i not in indices
                ]

    def send(self, to, msg):
        """Send a messge."""
        self._cleanup_pending_sends()

        req = COMM_WORLD.isend(msg, dest=to)
        self.pending_sends.append(req)

    def recv(self, wait=True):
        """Receive a message."""
        self._cleanup_pending_sends()

        if wait:
            return [self.msgq.get()]

        ret = []
        while True:
            try:
                m = self.msgq.get(block=False)
                ret.append(m)
            except queue.Empty:
                return ret

    def bcast_master(self, msg):
        """Do a broadcast on behalf of the master rank."""
        return COMM_WORLD.bcast(msg, root=MASTER_RANK)

    def finish(self):
        """Make sure all the pending messages are sent."""
        self.send(to=WORLD_RANK, msg=(Message.TERMINATE, None))

        if self.pending_sends:
            MPI.Request.Waitall(self.pending_sends)
            self.pending_sends = []

        self.receiver_thread.join()

    def _receiver_thread(self):
        """Code for the receiver thread."""
        while True:
            req = COMM_WORLD.irecv()
            type_, value = req.wait()
            if __debug__:
                log.debug("Rank %d received %s", WORLD_RANK, type_)
            if type_ == Message.TERMINATE:
                return

            self.msgq.put((type_, value))


class MPISimulator(Simulator):
    """MPI Simulator."""

    def __init__(self, *args, **kwargs):
        """Initialize."""
        super().__init__(*args, **kwargs)

        if WORLD_RANK == MASTER_RANK:
            self.is_master = True
        else:
            self.is_master = False

        # State variables used only by the master rank
        self.m_agent_population = None
        self.m_timestep_generator = None
        self.m_agent_distributor = None
        self.m_dead_agents = None

        # State variable used by both master and worker ranks
        self.w_agent_id_location = None
        self.w_local_agents = None
        self.w_received_messages = None
        self.w_incoming_messages = None
        self.w_workers_done = None

        # Run the actual inits
        if self.is_master:
            self._master_init()
        self._worker_init()

    def _master_init(self):
        """Initialize the master variables."""
        self.m_agent_population = self.agent_population_class(
            **self.agent_population_kwargs
        )

        self.m_timestep_generator = self.timestep_generator_class(
            **self.timestep_generator_kwargs
        )

        n_ranks = WORLD_SIZE
        rank_neighbors = [
            (rank_u, [(rank_v, 1.0) for rank_v in range(n_ranks) if rank_v != rank_u])
            for rank_u in range(n_ranks)
        ]
        self.m_agent_distributor = self.agent_distributor_class(
            self.m_agent_population,
            WORLD_SIZE,
            rank_neighbors,
            **self.agent_distributor_kwargs
        )

        self.m_dead_agents = []

    def _worker_init(self):
        """Initialize the worker variables."""
        self.w_agent_id_location = {}
        self.w_local_agents = {}

        # Messages received during last timestep
        self.w_received_messages = defaultdict(list)
        # Messages received during current timestep
        self.w_incoming_messages = defaultdict(list)

        self.w_workers_done = 0

    def run(self):
        """Run the simulation."""
        comm = Comm()
        if self.is_master:
            self._master_run(comm)
        else:
            self._worker_run(comm)
        comm.finish()

    def _master_run(self, comm):
        """Start the master thread."""
        log.info("Rank %d master is starting", WORLD_RANK)

        while True:
            # Get the current timestep
            timestep, timeperiod = self.m_timestep_generator.get_next_timestep()

            if timestep is None:
                # We are done
                comm.bcast_master((None, None, None, None))
                log.info("Rank %d master is exiting", WORLD_RANK)
                return

            # Get the new rank_agents_tuples
            rank_agent_constructor_kwargs_lists = self.m_agent_distributor.distribute(
                timestep, timeperiod
            )

            # Collect the dead agents
            dead_agents = self.m_dead_agents
            self.m_dead_agents = []

            # Distribute timestamp and agents
            comm.bcast_master(
                (timestep, timeperiod, rank_agent_constructor_kwargs_lists, dead_agents)
            )

            # Run the master's step
            self._worker_step_local_agents(
                comm,
                timestep,
                timeperiod,
                rank_agent_constructor_kwargs_lists,
                dead_agents,
            )

    def _worker_run(self, comm):
        """Start the worker thread."""
        log.info("Rank %d worker is starting", WORLD_RANK)

        while True:
            # Get the data from master
            (
                timestep,
                timeperiod,
                rank_agent_constructor_kwargs_lists,
                dead_agents,
            ) = comm.bcast_master(None)

            # We are done
            if timestep is None:
                log.info("Rank %d worker is exiting", WORLD_RANK)
                return

            if __debug__:
                log.debug("Rank %d start timestep %f (%s)", WORLD_RANK, timestep, timeperiod)

            # Run the worker's step
            self._worker_step_local_agents(
                comm,
                timestep,
                timeperiod,
                rank_agent_constructor_kwargs_lists,
                dead_agents,
            )

    def _worker_step_local_agents(
        self,
        comm,
        timestep,
        timeperiod,
        rank_agent_constructor_kwargs_lists,
        dead_agents,
    ):
        """Run a timestep."""
        # Note the start time
        rank_step_start_time = time()

        # Create new agents
        self._instantiate_agents(rank_agent_constructor_kwargs_lists)

        # Delete dead agents
        for agent_id in dead_agents:
            if agent_id in self.w_local_agents:
                del self.w_local_agents[agent_id]

        if __debug__:
            log.debug("Rank %d # local agents %d", WORLD_RANK, len(self.w_local_agents))

        # Update agent locations
        self._update_agent_id_rank(dead_agents, rank_agent_constructor_kwargs_lists)

        if __debug__:
            log.debug("Rank %d # total agents %d", WORLD_RANK, len(self.w_agent_id_location))

        # Iterate over the agents
        for agent_id, agent in self.w_local_agents.items():
            # Run the step on the agent
            self._worker_step_agent(comm, timestep, timeperiod, agent_id, agent)

            # If we have any incoming messages, process them
            for type_, value in comm.recv(wait=False):
                self._process_incoming_message(type_, value)

        # Log process's profile
        rank_step_time = time() - rank_step_start_time
        msg = (Message.RANK_STEP_PROFILE, (WORLD_RANK, timestep, rank_step_time))
        comm.send(to=MASTER_RANK, msg=msg)

        # Notify everyone that I am done
        msg = (Message.WORKER_DONE, WORLD_RANK)
        for rank in range(WORLD_SIZE):
            comm.send(to=rank, msg=msg)

        # While not all ranks are done
        while self.w_workers_done < WORLD_SIZE:
            for type_, value in comm.recv(wait=True):
                self._process_incoming_message(type_, value)

        # Reset the timestep variables
        self.w_received_messages = self.w_incoming_messages
        self.w_incoming_messages = defaultdict(list)
        self.w_workers_done = 0

        # Wait for everyone to finish
        COMM_WORLD.Barrier()

    def _worker_step_agent(self, comm, timestep, timeperiod, agent_id, agent):
        """Step through one local agent."""
        agent_step_start_time = time()
        received_msgs = self.w_received_messages[agent_id]

        # Do the step!
        sent_msgs = agent.step(timestep, timeperiod, received_msgs)

        # Send out the messages to other agents
        for dst_id, message in sent_msgs:
            dst_rank = self.w_agent_id_location[dst_id]
            if dst_rank == WORLD_RANK:
                self.w_incoming_messages[dst_id].append((agent_id, message))
            else:
                msg = (Message.AGENT_TO_AGENT, (agent_id, dst_id, message))
                comm.send(to=dst_rank, msg=msg)

        # Check if the agent is alive
        if not agent.is_alive():
            msg = (Message.AGENT_DIED, (agent_id, timestep, timeperiod))
            comm.send(to=MASTER_RANK, msg=msg)

        # Log the agent's performance
        agent_step_time = time() - agent_step_start_time
        memory_usage = agent.memory_usage()
        received_msg_sizes = [
            (src_id, len(message)) for src_id, message in received_msgs
        ]
        sent_msg_sizes = [(dst_id, len(message)) for dst_id, message in sent_msgs]
        msg = (
            Message.AGENT_STEP_PROFILE,
            (
                agent_id,
                timestep,
                agent_step_time,
                memory_usage,
                received_msg_sizes,
                sent_msg_sizes,
            ),
        )
        comm.send(to=MASTER_RANK, msg=msg)

    def _process_incoming_message(self, type_, value):
        """Process and incoming message."""
        if type_ == Message.AGENT_TO_AGENT:
            src_id, dst_id, message = value
            self.w_incoming_messages[dst_id].append((src_id, message))

        elif type_ == Message.AGENT_DIED:
            agent_id, timestamp, timeperiod = value
            self.m_dead_agents.append(agent_id)
            self.m_agent_distributor.agent_died(agent_id, timestamp, timeperiod)

        elif type_ == Message.AGENT_STEP_PROFILE:
            profile = value
            self.m_agent_distributor.agent_step_profile(*profile)

        elif type_ == Message.RANK_STEP_PROFILE:
            profile = value
            self.m_agent_distributor.rank_step_profile(*profile)

        elif type_ == Message.WORKER_DONE:
            self.w_workers_done += 1

        else:
            raise RuntimeError("Unexpected Message: %s" % Message(type_))

    def _update_agent_id_rank(self, dead_agents, rank_agent_constructor_kwargs_lists):
        """Update the agent location information."""
        for rank, agent_constructor_kwargs_list in rank_agent_constructor_kwargs_lists:
            for agent_id, _ in agent_constructor_kwargs_list:
                self.w_agent_id_location[agent_id] = rank
        for agent_id in dead_agents:
            del self.w_agent_id_location[agent_id]

    def _instantiate_agents(self, rank_agent_constructor_kwargs_lists):
        """Instantiate new agents."""
        for rank, agent_constructor_kwargs_list in rank_agent_constructor_kwargs_lists:
            if rank == WORLD_RANK:
                if __debug__:
                    log.debug("Rank %d # new local agents %d", WORLD_RANK, len(agent_constructor_kwargs_list))
                for agent_id, agent_constructor_kwargs in agent_constructor_kwargs_list:
                    kwargs = dict(self.agent_kwargs)
                    kwargs.update(agent_constructor_kwargs)
                    self.w_local_agents[agent_id] = self.agent_class(agent_id, **kwargs)
