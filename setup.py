from setuptools import setup

setup(
    name="indiana-chain",
    version="0.1.0",
    description="Indiana Chain core engine",
    py_modules=["indiana_core"],
    install_requires=["fastapi", "uvicorn", "torch", "numpy", "tokenizers", "watchdog"],
)
