# Copyright 2020 DeepMind Technologies Limited
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

"""Setuptools installation script."""

from setuptools import find_packages
from setuptools import setup

description = """dm_construction is a set of "Construction" tasks requiring
agents to stack blocks to achieve a goal, engaging compositional and physical
reasoning.
"""

setup(
    name="dm_construction",
    version="1.0.0.dev",
    description="DeepMind Construction tasks",
    long_description=description,
    author="DeepMind",
    license="Apache License, Version 2.0",
    keywords=["machine learning"],
    url="https://github.com/deepmind/dm_construction",
    packages=find_packages(),
    # Additional docker requirements should be installed separately (See README)
    install_requires=[
        "absl-py",
        "dm_env",
        "dm_env_rpc==1.0.2",
        "docker",
        "grpcio",
        "numpy",
        "portpicker",
        "scipy",
        "setuptools",
        "shapely",
    ],
    extras_require={"demos": ["matplotlib", "jupyter"]},
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Developers",
        "Intended Audience :: Education",
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Operating System :: POSIX :: Linux",
        "Operating System :: Microsoft :: Windows",
        "Operating System :: MacOS :: MacOS X",
        "Programming Language :: Python :: 3.6",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
)
