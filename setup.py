from setuptools import find_packages
from setuptools import setup

setup(
    name='interTwin',
    version='0.0.1',
    install_requires=['itwinai'],
    packages=find_packages('ai/src'),
    package_dir={'': 'ai/src'},
)