from setuptools import setup

setup(
    name="billgpt",
    packages=["billgpt"],  # Directly specify the package
    package_dir={"": "billgpt"},
    install_requires=[
        "torch",
        "numpy",
        "tqdm",
        "tiktoken",
    ],
)
