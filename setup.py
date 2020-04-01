import os
from os import path

from setuptools import find_packages, setup


def get_version():
    init_py_path = path.join(path.abspath(path.dirname(__file__)), "waymo_toolkit", "__init__.py")
    init_py = open(init_py_path, "r").readlines()
    version_line = [l.strip() for l in init_py if l.startswith("__version__")][0]
    version = version_line.split("=")[-1].strip().strip("'\"")

    # The following is used to build release packages.
    # Users should never use it.
    if os.getenv("BUILD_NIGHTLY", "0") == "1":
        from datetime import datetime

        date_str = datetime.today().strftime("%y%m%d")
        version = version + ".dev" + date_str

        new_init_py = [l for l in init_py if not l.startswith("__version__")]
        new_init_py.append('__version__ = "{}"\n'.format(version))
        with open(init_py_path, "w") as f:
            f.write("".join(new_init_py))
    return version


setup(
    name="waymo_toolkit",
    version=get_version(),
    author="DapengFeng",
    url="https://github.com/DapengFeng/waymo_toolkit",
    description="Extract the elements from waymo dataset",
    packages=find_packages(),
    python_requires=">=3.6",
    install_requires=["waymo-open-dataset-tf-2-1-0", "tabulate", "matplotlib",],
    extras_require={
        "all": ["shapely", "psutil"],
        "dev": ["flake8", "isort", "black==19.10b0", "flake8-bugbear", "flake8-comprehensions"],
    },
)
