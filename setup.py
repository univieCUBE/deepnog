import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name="deepnog",
    version="0.1.1",
    author="Lukas Gosch",
    author_email="gosch.lukas@gmail.com",
    description="Deep learning based command line tool for protein family "
                + "predictions.",
    keywords="deep learning bioinformatics neural networks protein families",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="",
    packages=setuptools.find_packages(),
    package_data={
        # Include parameters of NNs trained directly on a whole database
        # (currently not supported by DeepNOG)
        'deepnog': ['parameters/*/*.pth'],
        # Include parameters of NNs trained on specific levels/parts of a db
        'deepnog': ['parameters/*/*/*.pth'],
        # Include data and parameters for tests, edit if necessary!
        'tests': ['data/*.faa'],
        'tests': ['parameters/*.pth']
    },
    entry_points={
        'console_scripts': [
            'deepnog = deepnog.deepnog:main'
        ]
    },
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: BSD License",
        "Operating System :: OS Independent",
        "Topic :: Scientific/Engineering :: Artificial Intelligence"
    ],
    python_requires='>=3',
)