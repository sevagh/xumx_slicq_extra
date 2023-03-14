from setuptools import setup, find_packages
import os

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()


setup(
    name="xumx_slicq_v2",
    version=os.getenv("XUMX_SLICQ_V2_VERSION", "1.0.0"),
    author="Sevag Hanssian",
    author_email="sevagh@pm.me",
    url="https://github.com/sevagh/xumx-sliCQ-V2",
    description="V2 of my original sliCQT adaptation of Open-Unmix",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    python_requires=">=3.8",
    install_requires=[
        "torch>=1.13.1",
        "torchaudio>=0.13.1",
        "numpy",
        "tqdm",
        "norbert @ git+https://github.com/yoyololicon/norbert#egg=norbert",
    ],
    extras_require={
        "tensorrt": [
            "tensorrt"
        ],
        "onnx": [
            "onnxruntime"
        ],
        "devel": [
            "onnx",
            "tensorrt"
            "tensorboard",
            "torchinfo",
            "scikit-learn",
            "auraloss",
            "optuna",
            "optuna-dashboard",
            "matplotlib",
            "musdb==0.3.1",
            "museval==0.3.1",
        ]
    },
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3.8",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
