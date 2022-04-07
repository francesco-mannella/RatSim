from setuptools import setup
from setuptools.command.install import install as DistutilsInstall
from setuptools.command.egg_info import egg_info as EggInfo


class MyInstall(DistutilsInstall):
    def run(self):
        DistutilsInstall.run(self)


class MyEgg(EggInfo):
    def run(self):
        EggInfo.run(self)


setup(
    name="ratsim",
    version="0.1",
    author="Francesco Mannella",
    author_email="francesco.mannella@gmail.com",
    description="A 2D dynamical simulator of a rat agent with whiskers",
    url="https://github.com/francesco-mannella/RatSim",
    classifiers=[
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
    package_dir={"": "src"},
    packages=setuptools.find_packages(where="src"),
    python_requires=">=3.6",
    cmdclass={"install": MyInstall, "egg_info": MyEgg},
    install_requires=["gym", "box2d_py", "numpy", "matplotlib", "scikit-image"],
)
