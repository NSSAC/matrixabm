The Matrix ABM: An Agent Based Modeling Framework
=================================================

The Matrix ABM is an agent based modeling framework
for developing large scale social simulations
that can run on distributed computing platforms.

Full documentation
------------------

The full documentation is available at ``docs/source/index.rst``

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

Building the documentation from source
......................................

To build and view the documentation as HTML, execute the following commands::

    $ git clone https://github.com/NSSAC/matrixabm.git
    $ cd matrixabm
    $ pip install --editable .
    $ pip install -r dev_requirements.txt
    $ make -C docs
    $ <BROWSER-COMMAND> docs/build/html/index.html
