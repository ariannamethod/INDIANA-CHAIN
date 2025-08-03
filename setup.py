from setuptools import setup

setup(
    name="indiana-c",
    version="0.1.0",
    description="Indiana-C core engine",
    py_modules=["indiana_core"],
    install_requires=["fastapi", "uvicorn", "torch", "numpy", "tokenizers", "watchdog"],
)
