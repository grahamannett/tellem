from setuptools import setup

setup(
    name='tellem',
    version='0.0.1',
    author="graham annett",
    author_email="graham.annett@gmail.com",
    url="https://github.com/grahamannett/tellem",
    packages=['tellem'],
    install_requires=[
        'numpy',
        'torch',
        'pytest',
    ],
)