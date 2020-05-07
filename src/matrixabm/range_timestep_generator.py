"""A range timestep generator.

The `RangeTimestepGenerator` produces a given number of timesteps.
The realtime period represented by each timestep
genearted by `RangeTimestepGenerator` is one second.
"""

from .datatypes import Timestep
from .timestep_generator import TimestepGenerator


class RangeTimestepGenerator(TimestepGenerator):
    """Generate timesteps in the range [0, nsteps-1]."""

    def __init__(self, nsteps):
        """Initialize.

        Parameters
        ----------
        nsteps: int
            Number of timesteps to generate
        """
        self.nsteps = int(nsteps)
        self.step = 0

    def get_next_timestep(self):
        """Return the next timestep and associated timeperiods.

        Returns
        -------
        timestep: Timestep, optional
            The current timestep
        """
        if self.step == self.nsteps:
            return None

        timestep = Timestep(float(self.step), float(self.step), float(self.step + 1))
        self.step += 1

        return timestep
