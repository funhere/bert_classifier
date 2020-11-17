import sys
from io import open
from setuptools import setup, find_packages
import subprocess


with open("requirements.txt") as f:
    install_requires = f.read().strip().split("\n")

setup(
    name="bert_classifier",
    version="1.0.1",
    description="AI Library using BERT",
    license="Apache2",
    url="https://github.com/funhere/bert_classifier",
    long_description=open("README.md", "r", encoding="utf-8").read(),
    long_description_content_type="text/markdown",
    keywords="BERT NLP deep learning google",
    packages=find_packages(exclude=["*.tests", "*.tests.*", "tests.*", "tests"]),
    install_requires=install_requires,
    classifiers=[
        "Intended Audience :: Science/Research",
        "License :: OSI Approved :: Apache Software License",
        "Programming Language :: Python :: 3",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
    ],
    zip_safe=False,
)
