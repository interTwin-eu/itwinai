from setuptools import setup, find_packages

setup(
    name='itwinpreproc',
    description="Preprocessing module for interTwin",
    author="Matteo Bunino and Alexander Zoechbauer",
    author_email="matteo.bunino@cern.ch and alexander.zoechbauer@cern.ch",
    version='0.1',
    packages=find_packages('src'),
    package_dir={'': 'src'},
    entry_points={
        'console_scripts': [
            'itwinpreproc=itwinpreproc.cli:app'
        ]
    }
)
