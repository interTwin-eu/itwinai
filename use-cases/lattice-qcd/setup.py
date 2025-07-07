# Copyright (c) 2021-2023 Javad Komijani


from setuptools import setup


def readme():
    with open('README.rst') as f:
        return f.read()


packages = [
        'normflow',
        'normflow.action',
        'normflow.lib',
        'normflow.lib.combo',
        'normflow.lib.indexing',
        'normflow.lib.linalg',
        'normflow.lib.spline',
        'normflow.lib.stats',
        'normflow.mask',
        'normflow.mcmc',
        'normflow.nn',
        'normflow.nn.scalar',
        'normflow.prior'
        ]

package_dir = {
        'normflow': 'src',
        'normflow.action': 'src/action',
        'normflow.lib': 'src/lib',
        'normflow.lib.combo': 'src/lib/combo',
        'normflow.lib.indexing': 'src/lib/indexing',
        'normflow.lib.linalg': 'src/lib/linalg',
        'normflow.lib.spline': 'src/lib/spline',
        'normflow.lib.stats': 'src/lib/stats',
        'normflow.mask': 'src/mask',
        'normflow.mcmc': 'src/mcmc',
        'normflow.nn': 'src/nn',
        'normflow.nn.scalar': 'src/nn/scalar',
        'normflow.prior': 'src/prior'
        }

setup(name='normflow',
      version='1.1',
      description='Normalizing flow for generating lattice field configurations',
      packages=packages,
      package_dir=package_dir,
      url='http://github.com/jkomijani/normflow',
      author='Javad Komijani',
      author_email='jkomijani@gmail.com',
      license='MIT',
      install_requires=['numpy>=1.20', 'torch>=2.0'],
      zip_safe=False
      )
