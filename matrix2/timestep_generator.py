"""Timestep generators."""

from abc import ABC, abstractmethod
from dataclasses import dataclass

@dataclass
class Timestep:
    """A simulation timestep."""

    step: float  # logical time of the timestep
    start: float  # real start time of the timestep (inclusive)
    end: float  # real end time of the timestep (exclusive)

class TimestepGenerator(ABC):
    """Timestep genertor interface."""

    @abstractmethod
    def get_next_timestep(self):
        """Return the next timestep.

        Returns
        -------
            timestep: Timestep or None
                The current timestep.
        """

class RangeTimestepGenerator(TimestepGenerator):
    """Generate timesteps in the range [0, nsteps-1]."""

    def __init__(self, nsteps):
        """Initialize."""
        self.nsteps = int(nsteps)
        self.step = 0

    def get_next_timestep(self):
        """Return the next timestep and associated timeperiods."""
        if self.step == self.nsteps:
            return None

        timestep = Timestep(float(self.step), float(self.step), float(self.step + 1))
        self.step += 1

        return timestep
