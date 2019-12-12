The Matrix ABM: An Agent Based Modeling Framework
=================================================

The Matrix ABM is an agent based modeling framework
for developing large scale social simulations
that can run on distrubuted computing platforms.

Full documentation
------------------

The full documentation is available at ``docs/source/index.rst``

To view the documentation as HTML, compile it as follows::

    $ git clone https://github.com/NSSAC/matrixabm.git
    $ cd xactor
    $ pip install --editable .
    $ pip install -r dev_requirements.txt
    $ make -C docs
    $ <BROWSER-COMMAND> docs/build/html/index.html
