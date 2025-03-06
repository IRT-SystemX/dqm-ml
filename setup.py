from setuptools import setup
import os

version = "0.0.0"
try:
    version = os.environ['python_lib_version']
except KeyError:  # bamboo_buildNumber isn't defined, so we're not running in Bamboo
    pass

setup(
    version = version,
)
