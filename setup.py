from setuptools import setup, find_packages

setup(
    name="cf-voronoi",
    version="0.1.0",
    description="Compact prototype-based classifier with few-shot extension",
    long_description=open("README.md").read(),
    long_description_content_type="text/markdown",
    author="[Your Name]",
    license="MIT",
    packages=find_packages(),
    python_requires=">=3.9",
    install_requires=[
        "torch>=2.0",
        "numpy>=1.24",
        "scikit-learn>=1.3",
    ],
)
