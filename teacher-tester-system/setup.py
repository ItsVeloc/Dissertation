from setuptools import setup, find_packages

setup(
    name="teacher-tester-system",
    version="0.1.0",
    packages=find_packages(),
    install_requires=[
        "pydantic>=1.9.0",
        "pyyaml>=6.0",
        "pandas>=1.3.0",
    ],
    author="AI Teacher-Tester Team",
    author_email="example@example.com",
    description="A system with teacher and tester agents for adaptive learning",
    keywords="ai, education, machine learning",
    python_requires=">=3.7",
)