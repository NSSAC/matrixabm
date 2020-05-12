"""A greedy load balancer.

This load balancer on every balance step
greedily moves objects from the most loaded bucket
to the least loaded bucket.
This process continues until the imbalance
is below a tolerance threshold.
"""

import heapq
import random
import numpy as np

from .load_balancer import LoadBalancer

LAMBDA_A = 0.9
LAMBDA_B = 0.9
LAMBDA = 0.9
IMBALANCE_TOL = 0.05


class GreedyLoadBalancer(LoadBalancer):
    """Simple greedy load balancer."""

    def __init__(self, n_buckets):
        """Initialize."""
        super().__init__(n_buckets)
        self.bucket_objects = [set() for _ in range(self.n_buckets)]
        self.object_bucket = dict()

        self.object_la = {}
        self.object_lb = {}

        # The values of the following are only valid
        # after a call to balance
        self.object_load = {}
        self.bucket_load = np.zeros(self.n_buckets, dtype=np.float64)
        self.imbalance = 0.0

        self.new_objects = set()
        self.object_bucket_prev = dict()

    def reset(self):
        """Prepare the balancer for next balancing round."""
        self.new_objects.clear()
        self.object_bucket_prev.clear()

    def add_object(self, o, la, lb):
        """Add a new object."""
        b = random.randint(0, self.n_buckets - 1)

        self.bucket_objects[b].add(o)
        self.object_bucket[o] = b

        self.object_la[o] = la
        self.object_lb[o] = lb

        self.new_objects.add(o)

    def delete_object(self, o):
        """Remove an object."""
        b = self.object_bucket[o]

        del self.object_la[o]
        del self.object_lb[o]

        del self.object_bucket[o]
        self.bucket_objects[b].remove(o)

    def update_load(self, o, la, lb):
        """Set the load of the given objects."""
        p_la = self.object_la[o]
        p_lb = self.object_lb[o]

        la = (1 - LAMBDA_A) * p_la + LAMBDA_A * la
        lb = (1 - LAMBDA_B) * p_lb + LAMBDA_B * lb

        self.object_la[o] = la
        self.object_lb[o] = lb

    def _update_load(self):
        """Update the object and bucket load."""
        max_la = max(self.object_la.values())
        max_lb = max(self.object_lb.values())

        for o in self.object_bucket:
            la = self.object_la[o] / max_la
            lb = self.object_la[o] / max_lb
            l = (1 - LAMBDA) * la + LAMBDA * lb

            self.object_load[o] = l

        for i in range(self.n_buckets):
            self.bucket_load.fill(0.0)
            for o in self.bucket_objects[i]:
                self.bucket_load[i] += self.object_load[o]

    def _update_imbalance(self):
        """Check if there is a load imbalance."""
        min_load = self.bucket_load.min()
        max_load = self.bucket_load.max()
        sum_load = self.bucket_load.sum()

        self.imbalance = float((max_load - min_load) / sum_load)

    def _greedy_move(self,):
        """Greedily select agents to move from max loaded rank to min loaded rank."""
        src = np.argmax(self.bucket_load)
        dst = np.argmin(self.bucket_load)

        object_load_h = [(self.object_load[o], o) for o in self.bucket_objects[src]]
        heapq.heapify(object_load_h)

        moved = False
        while object_load_h:
            l, o = heapq.heappop(object_load_h)

            # If movement will still leave the src bucket
            # more or equally loaded than dst bucket,
            # then move the object
            if self.bucket_load[src] - l >= self.bucket_load[dst] + l:
                moved = True
                if o not in self.object_bucket_prev:
                    self.object_bucket_prev[o] = src

                self.bucket_load[src] -= l
                self.bucket_load[dst] += l
                self.bucket_objects[src].remove(o)
                self.bucket_objects[dst].add(o)
                self.object_bucket[o] = dst
            else:
                break

        return moved

    def balance(self):
        """Balance the load distribution in the buckets."""
        self._update_load()

        while True:
            self._update_imbalance()
            if self.imbalance < IMBALANCE_TOL:
                break

            moved = self._greedy_move()
            if not moved:
                break

        for o in list(self.object_bucket_prev.keys()):
            if o in self.new_objects:
                del self.object_bucket_prev[o]
            elif self.object_bucket_prev[o] == self.object_bucket[o]:
                del self.object_bucket_prev[o]

    def get_new_objects(self):
        """Return the bucket of the new objects."""
        ret = []
        for o in self.new_objects:
            b = self.object_bucket[o]
            ret.append((o, b))
        return ret

    def get_moving_objects(self):
        """Return the moving objects and their source and destination buckets."""
        ret = []
        for o in self.object_bucket_prev:
            srcb = self.object_bucket_prev[o]
            dstb = self.object_bucket[o]
            ret.append((o, srcb, dstb))
        return ret
