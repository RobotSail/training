import os
from setuptools import setup, find_packages

# read the requirements here
requirements = []
if os.path.exists('requirements.txt'):
    with open('requirements.txt', 'r', encoding='utf-8') as reqs:
        requirements = reqs.read().splitlines()

setup(
    name="instructlab-training",
    version="0.1.0",
    packages=find_packages(),
    include_package_data=True,
    install_requires=requirements,
)

