"""Timestep generator implementations."""

from abc import ABC, abstractmethod

class TimestepGenerator(ABC):
    """Interface for timestep generators."""

    @abstractmethod
    def get_next_timestep(self):
        """Return the next timestep and associated timeperiod.

        Returns
        -------
            timestep: float or None
                The current timestep.
            timeperiod: a tuple of floats (start, end) or None
                The realtime period corresponding to the given timestep.
        """


class RangeTimestepGenerator(TimestepGenerator):
    """Generate timesteps in the range [0, n_timesteps -1]."""

    def __init__(self, n_timesteps):
        """Initialize."""
        self.n_timesteps = int(n_timesteps)
        self.cur_timestep = 0

    def get_next_timestep(self):
        """Return the next timestep and associated timeperiods."""
        if self.cur_timestep == self.n_timesteps:
            timestep = None
            timeperiod = None
        else:
            timestep = float(self.cur_timestep)
            timeperiod = (timestep, timestep + 1.0)
            self.cur_timestep += 1

        return timestep, timeperiod
