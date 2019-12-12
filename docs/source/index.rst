The Matrix ABM: An Agent Based Modeling Framework
=================================================

Introduction
------------

The Matrix ABM is an agent based modeling framework
for developing large scale social simulations
that can run on distributed computing platforms.

Installation
------------

The Matrix ABM depends on mpi4py which requires a MPI implementation
and compiler tools be installed on the system.

Installing Open MPI and mpi4py inside a conda environment
.........................................................

To create a new virtual environment with conda,
have Anaconda/Miniconda setup on your system.
Installation instructions for Anaconda can be found
`here <https://docs.conda.io/projects/conda/en/latest/user-guide/install/>`_.
After installation of Anaconda/Miniconda
execute the following commands::

    $ conda create -n matrixabm -c conda-forge python=3 openmpi mpi4py

The above command creates a new conda environment called ``matrixabm``
with python, openmpi and mpi4py installed.

The following commands assume you are inside the above conda environment.

Installing Matrix ABM
.....................

One can use pip to install The Matrix ABM for PyPI as follows::

    $ pip install matrixabm

Dependencies of The Matrix ABM (including mpi4py) will be installed
as part of the above pip command.

To install The Matrix ABM from source::

    $ git clone https://github.com/NSSAC/matrixabm.git
    $ cd matrixabm
    $ pip install --editable .

Bluepill Simulation aka. Hello World
------------------------------------

The file ``tests/bluepillsim.py``,
in the source directory,
contains a simple simulation
written using the Matrix ABM.

The following command
executes the above simulation using two processes
on the current machine::

    $ mpiexec -n 2 python tests/bluepillsim.py bluepill_store.sqlite3

The above command with create the file ``bluepill_store.sqlite3``
in the current directory.

API
---

.. toctree::
   :maxdepth: 2

   api

Inter-Component Messages
------------------------

.. toctree::
   :maxdepth: 2

   message-sequence

Indices and tables
------------------

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`
