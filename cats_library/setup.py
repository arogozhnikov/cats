from __future__ import division, print_function, absolute_import
from setuptools import setup
import codecs

__author__ = 'Alex Rogozhnikov'


setup(
    name="cats",
    version='0.1.0',
    description="Algorithms for doing with categorical variables",
    long_description="Algorithms for doing with categorical variables",

    url='https://github.com/arogozhnikov/hep_ml',

    # Author details
    author='Alex Rogozhnikov',
    author_email='axelr@yandex-team.ru',

    # Choose your license
    # license='Apache 2.0',
    packages=['cats'],

    classifiers=[
        # Indicate who your project is intended for
        'Intended Audience :: Science/Research',

        # Pick your license as you wish (should match "license" above)
        'License :: OSI Approved :: Apache Software License',

        # Specify the Python versions you support here. In particular, ensure
        # that you indicate whether you support Python 2, Python 3 or both.
        'Programming Language :: Python :: 2.7 ',
        'Programming Language :: Python :: 3.4 ',
    ],

    # What does your project relate to?
    keywords='machine learning, supervised learning, '
             'uncorrelated methods of machine learning, high energy physics, particle physics',

    # List run-time dependencies here. These will be installed by pip when your project is installed.
    install_requires=[
        'numpy >= 1.9',
        'pythran',
    ],
)
