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
    install_requires=['imbalanced-learn',
                      'matplotlib',
                      'numpy',
                      'nutsflow==1.2.3',
                      'nutsml==1.2.2',
                      'pandas',
                      'scipy',
                      'pytorch-model-summary==0.1.2',
                      'scikit-learn',
                      'six',
                      'tabulate',
                      'tensorboard==2.3.0',
                      'torch==1.7.0',],
)

