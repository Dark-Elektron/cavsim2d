# -*- coding: utf-8 -*-
import os


from setuptools import setup, find_packages

with open(os.path.join(os.path.dirname(__file__), 'README.md')) as f:
    readme = f.read()

with open(os.path.join(os.path.dirname(__file__), 'LICENSE')) as f:
    license = f.read()

setup(
    name='cavsim2d',
    version='13.08.2024',
    description='A set of python codes for quick 2D axisymmetric rf structure analysis.',
    long_description=readme,
    author='Sosoho-Abasi Udongwo',
    author_email='numurho@gmail.com',
    url=r'https://github.com/Dark-Elektron/cavsim2d',
    license=license,
    python_requires='>=3.0, <4',
    install_requires=[
        'pandas>=2.0.0',
        'numpy>=1.14.5',
        'matplotlib>=2.2.0'
    ],
    packages=find_packages(exclude=('tests', 'docs')),
    classifiers=[
        "Programming Language :: Python",
        "Programming Language :: Python :: 3",
        "Development Status :: 4 - Beta",
        "Environment :: Other Environment",
        "Intended Audience :: Science/Research",
        "Natural Language :: English",
        "License :: OSI Approved :: GNU Library or Lesser General Public License (LGPL)",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering",
        "Topic :: Topic :: Scientific/Engineering :: Physics",
        ],
)
