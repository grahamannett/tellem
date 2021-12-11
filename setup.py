from setuptools import find_packages, setup

setup(
    name="tellem",
    version="0.0.1",
    author="graham annett",
    author_email="graham.annett@gmail.com",
    url="https://github.com/grahamannett/tellem",
    packages=find_packages(exclude=["tests", "*.tests", "*.tests.*", "tests.*"]),
    install_requires=[
        "numpy",
        "torch",
        "pytest",
    ],
)
