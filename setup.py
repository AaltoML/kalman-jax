# coding=utf-8
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Setup script for using kalman-jax.
This script will install kalman-jax as a Python module. Derived from 
https://github.com/google/dopamine/blobl/master/setup.py
"""

import pathlib
from setuptools import find_packages
from setuptools import setup

here = pathlib.Path(__file__).parent.resolve()

long_description = (here / 'README.md').read_text(encoding='utf-8')

install_requires = [
    'jax >= 0.1.67',
    'jaxlib >= 0.1.47',
    'matplotlib >= 3.1.3',
    'numpy >= 1.17.2',
    'scikit-learn >= 0.21.3',
    'scipy >= 1.4.1',
]

kalmanjax_description = (
    'Kalman-Jax: Approximate inference for Markov Gaussian processes using iterated Kalman filtering and smoothing.')

setup(
    name='kalman-jax',
    version='0.1.0',
    description=kalmanjax_description,
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/twkillian/kalman-jax',
    author='William Wilkinson',
    classifiers=[
        'Development Status :: 4 - Beta',

        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',

        'License :: OSI Approved :: Apache Software License',

        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.5',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3 :: Only',

        'Topic :: Scientific/Engineering',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
        'Topic :: Software Development',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',

    ],
    keywords='gaussian-processes, state-space-models, approximate-bayesian-inference, kalman-smoother, machine-learning, signal-processing',
    include_package_data=True,
    packages=find_packages(exclude=['docs']),
    package_data={'testdata': ['testdata/*.gin']},
    install_requires=install_requires,
    project_urls={  # Optional
        'Documentation': 'https://github.com/twkillian/kalman-jax',
        'Bug Reports': 'https://github.com/google/twkillian/kalman-jax',
        'Source': 'https://github.com/twkillian/kalman-jax',
    },
    license='Apache 2.0',
)