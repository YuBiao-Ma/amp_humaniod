from setuptools import find_packages
from distutils.core import setup

setup(
    name='HumanoidAmpLocomotion',
    version='1.0.0',
    author='AgiBot',
    license="BSD-3-Clause",
    packages=find_packages(),
    description='Isaac Gym environments for Legged Robots',
    # install_requires=['isaacgym',  # preview4
    #                   'tensorboard',
    #                   'numpy==1.23.5',
    #                   'opencv-python',
    #                   'mujoco==2.3.6',
    #                   'mujoco-python-viewer',
    #                   'matplotlib']
)
