# Distributed under the OSI-approved BSD 3-Clause License.  See accompanying
# file Copyright.txt or https://cmake.org/licensing for details.

cmake_minimum_required(VERSION 3.16.3)

# We name the project and the target for the ExternalProject_Add() call
# to something that will highlight to the user what we are working on if
# something goes wrong and an error message is produced.

project(ccd-populate NONE)

include(ExternalProject)
ExternalProject_Add(ccd-populate
                     "UPDATE_DISCONNECTED" "False" "GIT_REPOSITORY" "https://github.com/danfis/libccd.git" "GIT_TAG" "7931e764a19ef6b21b443376c699bbc9c6d4fba8" "GIT_SHALLOW" "FALSE" "PATCH_COMMAND"
                    SOURCE_DIR          "/home/kist-robot2/mujoco-2.3.1_source/mujoco-2.3.1/_deps/ccd-src"
                    BINARY_DIR          "/home/kist-robot2/mujoco-2.3.1_source/mujoco-2.3.1/_deps/ccd-build"
                    CONFIGURE_COMMAND   ""
                    BUILD_COMMAND       ""
                    INSTALL_COMMAND     ""
                    TEST_COMMAND        ""
                    USES_TERMINAL_DOWNLOAD  YES
                    USES_TERMINAL_UPDATE    YES
)
