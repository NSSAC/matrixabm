"""Timestep generator.

A Matrix simulation tracks both the logical time,
and the realtime period represented by that logical time.
Logical timesteps in the Matrix are allowed to represent
variable realtime periods.
The task of the `TimestepGenerator` is to determine
what realtime period the next logical timestep represents.
For this purpose, the timestep generator can consult
the local state store objects.

At the beginning of every timestep,
the Simulator actor calls the `get_next_timestep` method
of the timestep generator.
The method is supposed to return a `Timestep` object or None.
In case it returns None, the simulation ends.
"""

from abc import ABC, abstractmethod

from .datatypes import Timestep

class TimestepGenerator(ABC):
    """Timestep generator interface."""

    @abstractmethod
    def get_next_timestep(self):
        """Return the next timestep.

        Returns
        -------
        timestep : Timestep or None
            The current timestep
        """

class RangeTimestepGenerator(TimestepGenerator):
    """Generate timesteps in the range [0, nsteps-1].

    The `RangeTimestepGenerator` produces a given number of timesteps.
    The realtime period represented by each timestep
    genearted by `RangeTimestepGenerator` is one second.
    """

    def __init__(self, nsteps):
        """Initialize.

        Parameters
        ----------
        nsteps : int
            Number of timesteps to generate
        """
        self.nsteps = int(nsteps)
        self.step = 0

    def get_next_timestep(self):
        """Return the next timestep and associated timeperiods.

        Returns
        -------
        timestep : Timestep, optional
            The current timestep
        """
        if self.step == self.nsteps:
            return None

        timestep = Timestep(float(self.step), float(self.step), float(self.step + 1))
        self.step += 1

        return timestep
