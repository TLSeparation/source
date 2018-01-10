# -*- coding: utf-8 -*-
"""
Setup file for the tlseparation package.

@author: Matheus Boni Vicari (matheus.boni.vicari@gmail.com)
"""

from setuptools import setup, find_packages


def readme():
    with open('README.rst') as f:
        return f.read()

with open('requirements.txt') as f:
    required = f.read().splitlines()

setup(
    name="tlseparation",
    version="1.2.1.2",
    author='Matheus Boni Vicari',
    author_email='matheus.boni.vicari@gmail.com',
    packages=find_packages(),
    entry_points={
        'console_scripts': ['tlseparation=tlseparation.command_line:main']},
    url='https://github.com/mattbv/tlseparation',
    license='LICENSE.txt',
    description='Performs the wood/leaf separation from\
 3D point clouds generated using Terrestrial LiDAR\
 Scanners.',
    long_description=readme(),
    classifiers=['Programming Language :: Python',
                 'Topic :: Scientific/Engineering'],
    keywords='wood/leaf separation TLS point cloud LiDAR',
    install_requires=required,
    # ...
)
