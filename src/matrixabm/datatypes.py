"""Struct(ured) datatypes.

This module defines data container classes
commonly used in a Matrix simulation.
"""

from dataclasses import dataclass, field

@dataclass
class Timestep:
    """A simulation timestep.

    Matrix simulations are discrete time simulations.
    The Timestep structure is used to represent the logical step (or tick)
    as well as the real time period represented by by it.

    Attributes
    ----------
    step : float
        Logical (discrete) time of the timestep
    start : float
        Real start time of the timestep (inclusive)
    end : float
        Real end time of the timestep (exclusive)
    """

    step: float
    start: float
    end: float

@dataclass(init=False)
class Constructor:
    """A delayed object constructor.

    Attributes
    ----------
    cls : type
        The class to be used for creating the object
    args : list
        The positional arguments of the constructor
    kwargs : dict
        The keyword arguments of the constructor
    """

    cls: type
    args: list
    kwargs: dict

    def __init__(self, cls, *args, **kwargs):
        """Make the constructor.

        Parameters
        ----------
        cls : type
            The class to be used for creating the object
        *args : list
            Positional arguments of the constructor
        **kwargs : dict
            Keyword arguments of the constructor
        """
        self.cls = cls
        self.args = args
        self.kwargs = kwargs

    def construct(self):
        """Construct the object.

        Returns
        -------
        object
            The constructed object
        """
        return self.cls(*self.args, **self.kwargs)

@dataclass(init=False, order=True)
class StateUpdate:
    """A state update message.

    Attributes
    ----------
    store_name : str
        Name of the state store to which this update is to be applied
    order_key : str
        Key used to sort the updates before application
    method : str
        Method name on the state store that is used to apply this update
    args : list
        Positional arguments for the above method
    kwargs : dict
        Keyword arguments for the above method
    """

    store_name: str
    order_key: str

    method: str = field(compare=False)
    args: list = field(compare=False)
    kwargs: dict = field(compare=False)

    def __init__(self, store_name, order_key, method, *args, **kwargs):
        """Construct the update.

        Parameters
        ----------
        store_name : str
            Name of the state store to which this update is to be applied
        order_key : str
            Key used to sort the updates before application
        method : str
            Method name on the state store that is used to apply this update
        *args : list
            Positional arguments for the above method
        **kwargs : dict
            Keyword arguments for the above method
        """
        self.store_name = store_name
        self.order_key = order_key
        self.method = method
        self.args = args
        self.kwargs = kwargs

    def apply(self, store):
        """Apply the update to the store.

        Parameters
        ----------
        store : object
            The state store object
        """
        method = getattr(store, self.method)
        method(*self.args, **self.kwargs)
