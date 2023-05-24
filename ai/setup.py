from setuptools import setup, find_packages

setup(
    name="itwinai",
    description="AI and ML workflows module for interTwin",
    author="Matteo Bunino and Alexander Zoechbauer",
    author_email="matteo.bunino@cern.ch and alexander.zoechbauer@cern.ch",
    version="0.1",
    packages=find_packages("src"),
    package_dir={"": "src"},
    entry_points={
        "console_scripts": [
            "itwinai=itwinai.cli:app"
        ]
    },
    # Pip dependencies
    install_requires=[
        'jsonargparse[signatures]>=4.17.0'
    ]
)
