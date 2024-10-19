# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.16.3)

# We name the project and the target for the ExternalProject_Add() call
# to something that will highlight to the user what we are working on if
# something goes wrong and an error message is produced.

project(lodepng-populate NONE)

include(ExternalProject)
ExternalProject_Add(lodepng-populate
                     "UPDATE_DISCONNECTED" "False" "GIT_REPOSITORY" "https://github.com/lvandeve/lodepng.git" "GIT_TAG" "b4ed2cd7ecf61d29076169b49199371456d4f90b"
                    SOURCE_DIR          "/home/kist-robot2/mujoco-2.3.1_source/mujoco-2.3.1/_deps/lodepng-src"
                    BINARY_DIR          "/home/kist-robot2/mujoco-2.3.1_source/mujoco-2.3.1/_deps/lodepng-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
                    USES_TERMINAL_DOWNLOAD  YES
                    USES_TERMINAL_UPDATE    YES
)
