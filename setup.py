from setuptools import find_packages, setup

setup(
    name="indiana-c",
    version="0.1.0",
    description="Indiana-C core engine",
    packages=find_packages(),
    install_requires=["torch", "tokenizers"],
)
