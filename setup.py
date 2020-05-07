"""Setup."""

from setuptools import setup

package_name = "matrixabm"
description = "The Matrix ABM: An Agent Based Modeling Framework"

with open("README.rst", "r") as fh:
    long_description = fh.read()

classifiers = """
Intended Audience :: Developers
Intended Audience :: Science/Research
License :: OSI Approved :: MIT License
Operating System :: POSIX :: Linux
Programming Language :: C
Programming Language :: Python :: 3.7
Programming Language :: Python :: 3.8
Topic :: Scientific/Engineering
Topic :: System :: Distributed Computing
""".strip().split("\n")

setup(
    name=package_name,
    description=description,

    author="Parantapa Bhattacharya",
    author_email="pb+pypi@parantapa.net",

    long_description=long_description,
    long_description_content_type="text/x-rst",

    packages=[package_name],
    package_dir={'': 'src'},

    use_scm_version=True,
    setup_requires=['setuptools_scm'],

    install_requires=[
        "click",
        "click_completion",
        "logbook",
        "more-itertools",
        "python-dateutil",
        "sortedcontainers",
        "xactor",
        "numpy",
        "tensorboard",
        "tensorboardx",
    ],

    url="http://github.com/nssac/matrixabm",
    classifiers=classifiers
)
