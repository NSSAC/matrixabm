"""A random load balancer.

This load balancer completely ignores the load information.
Every new object is assigned to a random bucket.
The balance operation is a no op.
Agents never move from one bucker to another.
"""

import random

from .load_balancer import LoadBalancer


class RandomLoadBalancer(LoadBalancer):
    """A simple random load balancer."""

    def __init__(self, n_buckets):
        """Initialize."""
        super().__init__(n_buckets)

        self.bucket_objects = [set() for _ in range(self.n_buckets)]
        self.object_bucket = dict()

        self.new_objects = set()

    def reset(self):
        """Prepare the balancer for next balancing round."""
        self.new_objects.clear()

    def add_object(self, o, la, lb):
        """Add a new object.

        Parameters
        ----------
            o: string/int
                ID of the object
            la: float
                First component of object load (e.g. CPU usage)
            lb: float
                Second component of object load (e.g. Memory usage)
        """
        b = random.randint(0, self.n_buckets - 1)

        self.bucket_objects[b].add(o)
        self.object_bucket[o] = b

        self.new_objects.add(o)

    def delete_object(self, o):
        """Remove an object.

        Parameters
        ----------
            o: string/int
                ID of the object
        """
        b = self.object_bucket[o]

        del self.object_bucket[o]
        self.bucket_objects[b].remove(o)

    def update_load(self, o, la, lb):
        """Set the load of the given objects.

        Parameters
        ----------
            o: string/int
                ID of the object
            la: float
                First component of object load (e.g. CPU usage)
            lb: float
                Second component of object load (e.g. Memory usage)
        """

    def balance(self):
        """Balance the load distribution in the buckets."""

    def get_new_objects(self):
        """Return the bucket of the new objects.

        Returns
        -------
            List of two tuples [(o, b])
                o: string/int
                    ID of the object
                b: int
                    The bucket of the object
        """
        ret = []
        for o in self.new_objects:
            b = self.object_bucket[o]
            ret.append((o, b))
        return ret

    def get_moving_objects(self):
        """Return the moving objects and their source and dstination buckets.

        Returns
        -------
            List of two tuples [(o, srcb, dstb])
                o: string/int
                    ID of the object
                srcb: int
                    The source bucket of the object
                dstb: int
                    The destination bucket of the object
        """
        return []
