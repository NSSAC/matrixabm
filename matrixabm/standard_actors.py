"""Standard actor proxies."""

from xactor.mpi_actor import ActorProxy
from xactor.mpi_actor import MASTER_RANK, EVERY_RANK, WORLD_SIZE

# Standard Actor IDs
AID_MAIN = "main"
AID_POPULATION = "population"
AID_COORDINATOR = "coordinator"
AID_RUNNER = "runner"

# Standard Actor Proxyies
MAIN = ActorProxy(MASTER_RANK, AID_MAIN)
POPULATION = ActorProxy(MASTER_RANK, AID_POPULATION)
COORDINATOR = ActorProxy(MASTER_RANK, AID_COORDINATOR)
RUNNERS = [ActorProxy(rank, AID_RUNNER) for rank in range(WORLD_SIZE)]
EVERY_RUNNER = ActorProxy(EVERY_RANK, AID_RUNNER)
