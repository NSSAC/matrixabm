"""Agent runner."""

from time import perf_counter

import xactor as asys

from . import INFO_FINE, WORLD_SIZE

LOG = asys.getLogger(__name__)


class Runner:
    """Agent runner.

    The agent runner is responsible for
    running the step method of the agents to produce updates
    and sending them to the responsible state stores.
    It also sends the step profile info back to the coordinator.
    There is one runner actor per process/rank.

    Receives
    --------
    * `step` from Simulator
    * `create_agent*` from Coordinator
    * `create_agent_done` from Coordinator
    * `move_agent*` from Coordinator
    * `move_agent_done` from Coordinator
    * `receive_agent*` from Runner(s)
    * `receive_agent_done` from Runner(s)

    Sends
    -----
    * `handle_update*` to StateStore(s)
    * `handle_update_done` to StateStore(s)
    * `agent_step_profile` to Coordinator
    * `agent_step_profile_done` to Coordinator
    """

    def __init__(self, store_proxies, coordinator_aid, runner_aid):
        """Initialize the runner.

        Parameters
        ----------
        store_proxies : dict [str -> ActorProxy]
            Actor proxy objects to stores
        coordinator_aid : str
            ID of the coordinator actor
        runner_aid : str
            ID of the runner actors
        """
        self.local_agents = {}
        self.store_proxies = store_proxies
        self.coordinator_proxy = asys.ActorProxy(asys.MASTER_RANK, coordinator_aid)
        self.every_runner_proxy = asys.ActorProxy(asys.EVERY_RANK, runner_aid)
        self.runner_proxies = [asys.ActorProxy(rank, runner_aid) for rank in asys.ranks()]

        # Step variables
        self.timestep = None
        self.flag_create_agent_done = None
        self.flag_move_agents_done = None
        self.num_receive_agent_done = None

        self._prepare_for_next_step()

    def _prepare_for_next_step(self):
        """Reset the variables."""
        LOG.log(INFO_FINE, "Preparing for next step")

        self.timestep = None
        self.flag_create_agent_done = False
        self.flag_move_agents_done = False
        self.num_receive_agent_done = 0

    def _try_start_step(self):
        """Step through the local agents to produce updates."""
        LOG.log(
            INFO_FINE,
            "Can start step? (TS=%s,CAD=%s,MAD=%s,NRAD=%d/%d)",
            bool(self.timestep),
            self.flag_create_agent_done,
            self.flag_move_agents_done,
            self.num_receive_agent_done,
            WORLD_SIZE,
        )
        if self.timestep is None:
            return
        if not self.flag_create_agent_done:
            return
        if not self.flag_move_agents_done:
            return
        if self.num_receive_agent_done < WORLD_SIZE:
            return

        self.do_step()
        self._prepare_for_next_step()

    def do_step(self):
        """Do the actual stepping through over local agents to produce updates."""
        dead_agents = []

        # Step through the agents
        for agent_id, agent in self.local_agents.items():
            start_time = perf_counter()

            # Step through the agent
            updates = agent.step(self.timestep)
            memory_usage = agent.memory_usage()
            is_alive = agent.is_alive()
            if not is_alive:
                dead_agents.append(agent_id)

            # Send out the updates
            for update in updates:
                store_name = update.store_name
                store = self.store_proxies[store_name]
                store.handle_update(update)
            end_time = perf_counter()

            # Inform the coordinator
            self.coordinator_proxy.agent_step_profile(
                asys.current_rank(),
                agent_id=agent_id,
                step_time=(end_time - start_time),
                memory_usage=memory_usage,
                n_updates=len(updates),
                is_alive=is_alive,
            )

        # Tell stores that we are done for this step
        for store in self.store_proxies.values():
            store.handle_update_done(asys.current_rank())

        # Tell the coordinator we are done
        self.coordinator_proxy.agent_step_profile_done(asys.current_rank())

        # Flush out any pending messages
        asys.flush()

        # Delete any dead agents
        for agent_id in dead_agents:
            del self.local_agents[agent_id]


    def step(self, timestep):
        """Respond to the step signal from the simulator.

        Parameters
        ----------
        timestep : Timestep
            The current (to start) timestep
        """
        assert self.timestep is None

        self.timestep = timestep
        self._try_start_step()

    def create_agent(self, agent_id, constructor):
        """Create an agent locally.

        Parameters
        ----------
        agent_id : str
            ID of the to be created agent
        constructor : Constructor
            Constructor to create the agent
        """
        if __debug__:
            if agent_id in self.local_agents:
                raise RuntimeError(
                    "Can't create agent; agent %s already exists" % agent_id
                )

        self.local_agents[agent_id] = constructor.construct()

    def create_agent_done(self):
        """Respond to create event done message from coordinator."""
        assert not self.flag_create_agent_done

        self.flag_create_agent_done = True
        self._try_start_step()

    def move_agent(self, agent_id, dst_rank):
        """Send local agents to destination ranks.

        Parameters
        ----------
        agent_id : str
            ID of the agent to be moved
        dst_rank : int
            Destination rank of the agent
        """
        if __debug__:
            if agent_id not in self.local_agents:
                raise RuntimeError(
                    "Can't send agent; agent %s doesn't exist" % agent_id
                )

        agent = self.local_agents[agent_id]
        self.runner_proxies[dst_rank].receive_agent(agent_id, agent)

        del self.local_agents[agent_id]

    def move_agent_done(self):
        """Respond to move agents done message from coordinator."""
        assert not self.flag_move_agents_done
        self.flag_move_agents_done = True

        self.every_runner_proxy.receive_agent_done(asys.current_rank())
        self._try_start_step()

    def receive_agent(self, agent_id, agent):
        """Receive an agent from another worker process.

        Parameters
        ----------
        agent_id : str
            ID of the incoming agent
        agent : Agent
            The actual agent itself
        """
        if __debug__:
            if agent_id in self.local_agents:
                raise RuntimeError(
                    "Can't receive agent; agent %s already exists" % agent_id
                )

        self.local_agents[agent_id] = agent

    def receive_agent_done(self, rank):
        """Respond to receive agent done message from other agent steppers.

        Parameters
        ----------
        rank : int
            Rank of the agent runner
        """
        assert self.num_receive_agent_done < WORLD_SIZE
        if __debug__:
            LOG.debug("Runner on %d has finished sending agents", rank)

        self.num_receive_agent_done += 1
        self._try_start_step()
