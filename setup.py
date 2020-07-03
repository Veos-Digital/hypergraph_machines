# This file is part of Discrete and continuous learning machines.
# Copyright (C) 2020- Mattia G. Bergomi, Patrizio Frosini, Pietro Vertechi
#
# Discrete and continuous learning machines is free software: you can
# redistribute it and/or modify it under the terms of the GNU General Public
# License as published by the Free Software Foundation, either version 3 of the
# License, or (at your option) any later version.
#
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details. In addition, we require
# derivatives or applications to acknowledge the authors by citing [1].
#
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <https://www.gnu.org/licenses/>.
#
# For more information please use the tools available at
# https://gitlab.com/mattia.bergomi.
#
# [1]

import os
import sys
from distutils.sysconfig import get_python_lib
from setuptools import find_packages, setup
import subprocess

CURRENT_PYTHON = sys.version_info[:2]
REQUIRED_PYTHON = (3, 6)
if CURRENT_PYTHON < REQUIRED_PYTHON:
    sys.stderr.write("""
==========================
Unsupported Python version
==========================
This version of hypergraph_machines requires Python {}.{}, but you're trying to
install it on Python {}.{}.
""".format(*(REQUIRED_PYTHON + CURRENT_PYTHON)))
    sys.exit(1)

requirements = ['torch>=1.1.0',
                'torchvision>=0.3.0',
                'numpy>=1.15.0',
                'networkx>=2.4',
                'seaborn>=0.9.0',
                'matplotlib>=2.2.2'
                ]
EXCLUDE_FROM_PACKAGES = []

setup(
    name='hypergraph_machines',
    version='0.0.0-prealpha',
    python_requires='>={}.{}'.format(*REQUIRED_PYTHON),
    url='',
    author='',
    author_email='',
    description=(''),
    license='GNU General Public License v3 or later (GPLv3+)',
    packages=find_packages(exclude=EXCLUDE_FROM_PACKAGES),
    include_package_data=True,
    install_requires=requirements,
    entry_points={},
    zip_safe=False,
    classifiers=[
        'Development Status :: 2 - Pre-Alpha',
        'Environment :: Console',
        'Intended Audience :: Science/Research',
        'Intended Audience :: Developers',
        'License :: OSI Approved :: GNU General Public License v3 or later (GPLv3+)',
        'Operating System :: OS Independent',
        'Programming Language :: Python',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Topic :: Scientific/Engineering'
        'Topic :: Scientific/Engineering :: Machine Learning',
        'Topic :: Scientific/Engineering :: Machine cognition',
    ],
)
