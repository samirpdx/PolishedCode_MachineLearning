from setuptools import setup, find_packages

PACKAGES = find_packages()

opts = dict(name='PolishedCode_MachineLearning',
            maintainer='Samir Patel',
            description='Polished Code Assignment - Coordinate Descent using Elastic Net',
            url='https://github.com/samirpdx/PolishedCode_MachineLearning',
            author='Samir Patel',
            packages=PACKAGES)

if __name__ == '__main__':
    setup(**opts)