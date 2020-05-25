from setuptools import setup, find_packages

exec(open("bsrs/version.py").read())

setup(
    name="bsrs",
    version=__version__,  # noqa
    packages=find_packages(),
    install_requires=[
        "numpy",
        "scipy",
        "tqdm",
        "sklearn",
        "matplotlib",
        "opencv-python",
    ],
)
