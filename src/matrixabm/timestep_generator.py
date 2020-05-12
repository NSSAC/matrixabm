"""Timestep generator interface.

A Matrix simulation tracks both the logical time,
and the realtime period represented by that logical time.
Logical timesteps in the Matrix are allowed to represent
variable realtime periods.
The task of the `TimestepGenerator` is to determine
what realtime period the next logical timestep represents.
For this purpose, the timestep generator can consult
the local state store objects.

At the beginning of every timestep,
the Main actor calls the `get_next_timestep` method
of the timestep generator.
The method is supposed to return a `Timestep` object or None.
In case it returns None, the simulation ends.
"""

from abc import ABC, abstractmethod


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
