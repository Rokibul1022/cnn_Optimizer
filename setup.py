from setuptools import setup, find_packages

setup(
    name="cnn-optimizer",
    version="1.0.0",
    packages=find_packages(),
    install_requires=[
        "torch>=2.0.0",
        "torchvision>=0.15.0",
        "numpy>=1.24.0",
        "pillow>=9.5.0",
        "matplotlib>=3.7.0",
        "tqdm>=4.65.0",
        "pycocotools>=2.0.6",
        "tabulate>=0.9.0",
    ],
    python_requires=">=3.8",
)
