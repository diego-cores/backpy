# -*- coding: utf-8 -*-

from setuptools import setup, find_packages

with open("README.md", encoding="utf-8") as file:
    desc = file.read()

setup(
    name='backpyf',
    version='0.9.71b4',
    packages=find_packages(),
    package_data={"backpyf": ["assets/*"]},
    description='''BackPy is a library made in python for back testing in 
    financial markets. Read Risk_notice.txt and LICENSE.''',
    long_description=desc,
    long_description_content_type='text/markdown',
    author='Diego',
    url='https://github.com/Diego-Cores/BackPy',
    license='MIT',
    classifiers=[
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
    ],
    python_requires=">=3.9",
    install_requires=[
        'pandas>=2.3.0',
        'numpy>=2.3.2',
        'matplotlib>=3.7.5',
        'pillow>=11.3.0',
    ],
    extras_require={
        'optional': [
            'yfinance==0.2.50',
            'binance-connector==3.10.0',
            'binance-futures-connector==4.1.0']
    }
)
