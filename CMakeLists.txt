cmake_minimum_required(VERSION 3.15)
project(SBM LANGUAGES CXX)

# ================================================================
# Use LLVM on macOS (required for OpenMP)
# ================================================================
if(APPLE)
    set(CMAKE_C_COMPILER "/opt/homebrew/opt/llvm/bin/clang")
    set(CMAKE_CXX_COMPILER "/opt/homebrew/opt/llvm/bin/clang++")
endif()

# ================================================================
# Python + NumPy
# ================================================================
find_package(Python REQUIRED COMPONENTS Interpreter Development.Module Development.Embed NumPy)

# ================================================================
# OpenMP
# ================================================================
find_package(OpenMP REQUIRED)

# ================================================================
# MonteCarlo_Potts
# ================================================================
add_library(MonteCarlo_Potts MODULE
    src/SBM/MonteCarlo/MCMC_Potts/MonteCarlo_PottsMod.cpp
)

target_include_directories(MonteCarlo_Potts PRIVATE
    ${Python_INCLUDE_DIRS}
    ${Python_NumPy_INCLUDE_DIRS}
)

target_link_libraries(MonteCarlo_Potts PRIVATE
    Python::Python
    OpenMP::OpenMP_CXX
)

set_target_properties(MonteCarlo_Potts PROPERTIES PREFIX "")

install(TARGETS MonteCarlo_Potts
    LIBRARY DESTINATION SBM/MonteCarlo/MCMC_Potts
)

# ================================================================
# MonteCarlo_PottsProf
# ================================================================
add_library(MonteCarlo_PottsProf MODULE
    src/SBM/MonteCarlo/MCMC_PottsProf/MonteCarlo_PottsProfMod.cpp
)

target_include_directories(MonteCarlo_PottsProf PRIVATE
    ${Python_INCLUDE_DIRS}
    ${Python_NumPy_INCLUDE_DIRS}
)

target_link_libraries(MonteCarlo_PottsProf PRIVATE
    Python::Python
    OpenMP::OpenMP_CXX
)

set_target_properties(MonteCarlo_PottsProf PROPERTIES PREFIX "")

install(TARGETS MonteCarlo_PottsProf
    LIBRARY DESTINATION SBM/MonteCarlo/MCMC_PottsProf
)
