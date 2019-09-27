"""Setup."""

from setuptools import setup

package_name = "matrix2"
description = "The Matrix v2: An Agent Based Modeling Framework"

with open("README.md", "r") as fh:
    long_description = fh.read()

setup(
    name=package_name,
    description=description,

    author="Parantapa Bhattacharya",
    author_email="pb+pypi@parantapa.net",

    long_description=long_description,
    long_description_content_type="text/markdown",

    packages=[package_name],

    use_scm_version=True,
    setup_requires=['setuptools_scm'],

    install_requires=[
        "click",
        "click_completion",
        "logbook",
        "toml",
        "more-itertools",
        "sortedcontainers"
        # "mpi4py",
    ],

    url="http://github.com/nssac/matrix2",
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        # "Operating System :: OS Independent",
    ),
)
