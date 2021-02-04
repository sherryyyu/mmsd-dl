from setuptools import setup, find_packages

import mmsddl

setup(
    name='mmsddl',
    version=mmsddl.__version__,
    url='https://github.com/sherryyyu/mmsd-dl',
    author='Shuang Yu, Fatemeh Jalali, Jianbin Tang',
    author_email='sherry.shuang.yu@gmail.com',
    description='Exploring CNN networks for multi-modal seizure detection',
    packages=find_packages(),
    install_requires=['numpy >= 1.19.4', 'torch >= 1.7.0', 'nutsflow == 1.1.0', 'nutsml == 1.1.0'],
)

