##
#  CMake script for the plasticity_model and Von_Mises_algorithm programs
##

cmake_minimum_required(VERSION 3.13.4)

# Set the name of the project
project(MEC5068Z_AGB)

# Find the deal.II library
find_package(deal.II 9.6.0 REQUIRED
        HINTS ${deal.II_DIR} ${DEAL_II_DIR} ../ ../../ $ENV{DEAL_II_DIR}
)

# Initialize deal.II variables
deal_ii_initialize_cached_variables()

# Check for required dependencies
if(NOT DEAL_II_WITH_MPI OR NOT DEAL_II_WITH_P4EST OR NOT DEAL_II_WITH_TRILINOS)
    message(FATAL_ERROR "
Error! This project requires a deal.II library that was configured with the following options:
    DEAL_II_WITH_MPI = ON
    DEAL_II_WITH_P4EST = ON
    DEAL_II_WITH_TRILINOS = ON
However, the deal.II library found at ${DEAL_II_PATH} was configured with these options:
    DEAL_II_WITH_MPI = ${DEAL_II_WITH_MPI}
    DEAL_II_WITH_P4EST = ${DEAL_II_WITH_P4EST}
    DEAL_II_WITH_TRILINOS = ${DEAL_II_WITH_TRILINOS}
This conflicts with the requirements."
    )
endif()

# Define the output that should be cleaned
set(CLEAN_UP_FILES *.vtu *.pvtu *.visit)

# List all your targets
set(TARGETS plasticity_model Von_Mises_algorithm)

# Loop over each target and set it up
foreach(TARGET ${TARGETS})
    # Specify the source files for each target
    if(${TARGET} STREQUAL "plasticity_model")
        set(SOURCES plasticity_model.cc)
    elseif(${TARGET} STREQUAL "Von_Mises_algorithm")
        set(SOURCES Von_Mises_algorithm.cc)
    endif()

    # Create the executable
    add_executable(${TARGET} ${SOURCES})

    # Link with deal.II
    deal_ii_setup_target(${TARGET})
endforeach()
