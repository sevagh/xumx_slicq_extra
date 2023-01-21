from setuptools import setup, find_packages

xumx_version = "0.1.0"

with open("README.md", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="xumx_slicq_v2",
    version=xumx_version,
    author="Sevag Hanssian",
    author_email="sevagh@pm.me",
    url="https://github.com/sevagh/xumx-sliCQ-V2",
    description="V2 of my original sliCQT adaptation of Open-Unmix",
    long_description=long_description,
    long_description_content_type="text/markdown",
    license="MIT",
    python_requires=">=3.8",
    install_requires=["numpy", "torchaudio>=0.8.0", "torch>=1.8.0", "tqdm"],
    extras_require={
        "tests": [
            "pytest",
            "musdb>=0.4.0",
            "museval>=0.4.0",
            "onnx",
            "tqdm",
        ],
        "stempeg": ["stempeg"],
        "evaluation": ["musdb>=0.4.0", "museval>=0.4.0"],
    },
    entry_points={"console_scripts": ["xumx-slicq-v2=xumx_slicq_v2.cli:separate_main", "xumx-slicq-v2-train=xumx_slicq_v2.train:train_main"]},
    packages=find_packages(),
    include_package_data=True,
    classifiers=[
        "Development Status :: 4 - Beta",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)
