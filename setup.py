import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="DeepNOG",
    version="0.1.0",
    author="Lukas Gosch",
    author_email="gosch.lukas@gmail.com",
    description="Deep learning based command line tool for protein family "
                + "predictions.",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://git.csb.univie.ac.at/gosch/deepnog",
    packages=setuptools.find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    python_requires='>=3',
)