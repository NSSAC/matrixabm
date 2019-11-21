"""Setup."""

from setuptools import setup

package_name = "matrixabm"
description = "The Matrix ABM: An Agent Based Modeling Framework"

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
        "xactor",
        "numpy",
        "tensorboard",
        "tensorboardx",
    ],

    url="http://github.com/nssac/matrixabm",
    classifiers=(
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
    ),
)
