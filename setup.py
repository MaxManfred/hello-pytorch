# Copyright (c) Konica Minolta Business Solutions. All Rights Reserved.
# Unauthorized copying of this file, via any medium is strictly prohibited.
# Proprietary and confidential.
# Written by Massimo Manfredino <massimo.manfredino@gmail.com>

from setuptools import setup, find_packages

__version__ = "0.0.1"

TESTS_REQUIRE = [
    'unittest',
]

with open('requirements.txt') as f:
    requirements = f.read().splitlines()

setup(
    name='hello-pytorch',
    version=__version__,
    python_requires='>=3.6.0',
    description='Pytorch tutorials',
    author='Massimo Manfredino',
    author_email='massimo.manfredino@gmail.com',
    packages=find_packages(),
    namespace_packages=['ai'],
    install_requires=requirements,
    extras_require={'test': TESTS_REQUIRE},
    include_package_data=True
)
