import setuptools

with open("README.md", "r") as fh:
    long_description = fh.read()

setuptools.setup(
    name = "mlearn",
    version = "0.0.1",
    author = "Zeerak Waseem",
    author_email = "zeerak.w@gmail.com",
    decription = "A package to contain machine learning pipelines for python.",
    long_description = long_description,
    long_description_content_type = "text/markdown",
    url="git@github.com:ZeerakW/mlearn.git",
    packages = setuptools.find_packages(),
    classifiers=[
        "Programming Languge :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: Unix"
    ],
    python_requires='>3.6',
    install_requires=[
        "Cython==0.29.21",
        "numpy==1.18.2",
        "tqdm==4.36.1",
        "spacy==2.2.2",
        "scikit-learn==0.23.1",
        "torchtestcase",
        "torchtext==0.6.0",
        "torch==1.4.0",
        'bpemb',
        'ekphrasis',
        'wandb',
        "en_core_web_sm @ https://github.com/explosion/spacy-models/releases/download/en_core_web_sm-2.2.5/en_core_web_sm-2.2.5.tar.gz#egg=en_core_web_sm"
    ],
    dependency_links=["https://download.pytorch.org/whl/cu90/torch_stable.html"]
    )
