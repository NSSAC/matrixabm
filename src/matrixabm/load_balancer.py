"""Load balancer interface.

The `LoadBalancer` encapsulates the agent load balancing logic.
"""

from abc import ABC, abstractmethod

class LoadBalancer(ABC):
    """Load Balancer interface.

    Attributes
    ----------
    n_buckets : int
        Number of buckets
    """

    def __init__(self, n_buckets):
        """Initialize."""
        self.n_buckets = int(n_buckets)

    @abstractmethod
    def reset(self):
        """Prepare the balancer for next balancing round."""

    @abstractmethod
    def add_object(self, o, la, lb):
        """Add a new object.

        Parameters
        ----------
        o : str or int
            ID of the object
        la : float
            First component of object load (e.g. CPU usage)
        lb : float
            Second component of object load (e.g. Memory usage)
        """

    @abstractmethod
    def delete_object(self, o):
        """Remove an object.

        Parameters
        ----------
        o : str or int
            ID of the object
        """

    @abstractmethod
    def update_load(self, o, la, lb):
        """Update the load of the given object.

        Parameters
        ----------
        o : str or int
            ID of the object
        la : float
            First component of object load (e.g. CPU usage)
        lb : float
            Second component of object load (e.g. Memory usage)
        """

    @abstractmethod
    def balance(self):
        """Balance the load distribution in the buckets."""

    @abstractmethod
    def get_new_objects(self):
        """Return the bucket of the new objects.

        Returns
        -------
        list of two tuples [(o, b])
            o : str or int
                ID of the object
            b : int
                The bucket of the object
        """

    @abstractmethod
    def get_moving_objects(self):
        """Return the moving objects and their source and dstination buckets.

        Returns
        -------
        list of three tuples [(o, srcb, dstb])
            o : str or int
                ID of the object
            srcb : int
                The source bucket of the object
            dstb : int
                The destination bucket of the object
        """
