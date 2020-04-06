import pathlib
from setuptools import find_packages
from distutils.core import setup

HERE = pathlib.Path(__file__).parent

README = (HERE / "README.md").read_text("utf-8")

setup(
    name="cv19index",
    version="1.1.1",
    description="COVID-19 Vulnerability Index",
    long_description=README,
    long_description_content_type="text/markdown",
    url="https://cv19index.com",
    author="Dave Decaprio",
    author_email="dave.decaprio@closedloop.ai",
    license="BSD",
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Healthcare Industry",
        "Intended Audience :: Science/Research",
        "Intended Audience :: Developers",
        "Natural Language :: English",
        "Framework :: Jupyter",
        "License :: OSI Approved :: BSD License",
        "Programming Language :: Python :: 3.6",
        "Programming Language :: Python :: 3.7",
    ],
    packages=find_packages(),
    include_package_data=True,
    zip_safe=False,
    package_data={
        "cv19index": [
            "resources/xgboost/input.csv.schema.json",
            "resources/xgboost/model.pickle",
            "resources/xgboost_all_ages/input.csv.schema.json",
            "resources/xgboost_all_ages/model.pickle",
            "resources/demo.schema.json",
            "resources/claims.schema.json",
            "resources/ccsrEdges.txt",
            "resources/ccsrNodes.txt",
        ]
    },
    entry_points={
        "console_scripts": [
            "cv19index=cv19index.predict:main",
            # serve is required to be exposed by the sagemaker API.
            "serve=cv19index.server:sagemaker_serve",
        ]
    },
    install_requires=[
        "numpy>=1.17.4",
        "pandas>=0.23.4",
        "setuptools>=40.2.0",
        "shap>=0.33",
        "xgboost>=1.0.1",
        "flask>=1.1.1",
        "regex>=2020.2.20",
        "xlrd >= 0.9.0"
    ],
)
