from setuptools import setup, find_packages
import re


req_prsing = re.compile(r"([^# \n]*)([# \n].*)?")
with open('requirements.txt') as req:
    requirements = [m.groups()[0] for m in map(req_prsing.match, req) if m.groups()[0]]


setup(
    name='spyne',
    version='0.0.1',
    description='A minimalist deep learning famework written "purely" in python.',
    url='https://github.com/bwhitesell/SpyNE',
    author='Ben Whitesell',
    author_email='whitesell.ben@gmail.com',
    license='BSD (3-clause)',
    keywords='deep learning machine learning regression classification neural networks',
    packages=find_packages(),
    install_requires=requirements,
    classifiers=[
            'Development Status :: 1 - Beta',
            'Intended Audience :: Developers',
            'Intended Audience :: Data Scientists',
            'License :: BSD 3-Clause "New" or "Revised" License',
            'Programming Language :: Python :: 3',
            'Programming Language :: Python :: 3.6',
            'Programming Language :: Python :: 3.7',
        ],
    zip_safe=False,

)