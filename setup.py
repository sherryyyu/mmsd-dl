from setuptools import setup, find_packages

import mmsddl

setup(
    name='mmsddl',
    version=mmsddl.__version__,
    url='https://github.ibm.com/aur-bic/mmsd-cnn',
    author='Shuang Yu, Fatemeh Jalali, Jianbin Tang',
    author_email='shuang.yu@ibm.com',
    description='Exploring CNN networks for multi-modal seizure detection',
    packages=find_packages(),
    install_requires=[],  # e.g. ['numpy >= 1.11.1', 'matplotlib >= 1.5.1']
)

