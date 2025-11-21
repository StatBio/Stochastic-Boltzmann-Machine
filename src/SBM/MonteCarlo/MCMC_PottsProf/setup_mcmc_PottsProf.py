import os
from setuptools import setup, Extension
import numpy as np

os.environ["CC"]="gcc"
os.environ["CXX"]="g++"
sfc_module = Extension("MonteCarlo_PottsProf",
                sources = ["src/SBM/MonteCarlo/MCMC_PottsProf/MonteCarlo_PottsProfMod.cpp"],
				include_dirs = [np.get_include()],
                extra_compile_args = ["-DNDEBUG", "-O3", "-std=c++17", "-fopenmp"],
                extra_link_args = ['-lgomp']
)

setup(
    name="MonteCarlo_PottsProf",
    version='1.0',
    description="Python Package with MonteCarlo C++ extension",
    ext_modules=[sfc_module]
)