from setuptools import setup, find_packages

setup(
    name="docking-consistency-tool",
    version="1.0.0",
    description="Automated docked pose consistency analysis pipeline",
    author="Computational Chemistry Lab",
    python_requires=">=3.9",
    packages=find_packages(),
    entry_points={
        "console_scripts": [
            "pose_consistency=pose_consistency:main",
        ],
    },
    install_requires=[
        "numpy>=1.24",
        "pandas>=2.0",
        "scipy>=1.10",
        "scikit-learn>=1.3",
        "matplotlib>=3.7",
        "seaborn>=0.12",
        "plotly>=5.15",
        "tqdm>=4.65",
        "jinja2>=3.1",
    ],
)
