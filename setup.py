
from setuptools import setup, find_packages

setup(
    name='backpyf',
    version='0.9.2.034',
    packages=find_packages(),
    description='''BackPy is a library made in python for back testing in 
    financial markets. Read Risk_notice.txt and LICENSE.''',
    long_description=open('README.md').read(),
    long_description_content_type='text/markdown',
    author='Diego',
    url='https://github.com/Diego-Cores/BackPy',
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    install_requires=[
        'pandas>=1.3.0',
        'numpy>=1.16.5',
        'matplotlib>=3.7.5'
    ],
    extras_require={
        'optional': ['yfinance>=0.2.36']
    }
)
