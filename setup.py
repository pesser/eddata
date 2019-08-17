from setuptools import setup, find_packages

setup(
    name="eddata",
    version="0.1",
    description="A collection of datasets.",
    packages=find_packages(),
    install_requires=["fastnumbers", "numpy", "pathlib", "tqdm"],
    zip_safe=False,
)
