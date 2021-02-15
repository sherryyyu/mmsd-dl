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
    install_requires=['imbalanced-learn==0.7.0',
                      'matplotlib==3.3.1',
                      'numpy==1.20.1',
                      'nutsflow==1.2.3',
                      'nutsml==1.2.2',
                      'pandas==1.1.1',
                      'scipy==1.5.4',
                      'pytorch-model-summary==0.1.2',
                      'scikit-learn==0.23.2',
                      'six==1.15.0',
                      'tabulate==0.8.7',
                      'tensorboard==2.3.0',
                      'torch==1.7.0',],
)

