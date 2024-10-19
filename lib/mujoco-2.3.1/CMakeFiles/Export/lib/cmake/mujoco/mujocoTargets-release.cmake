#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "mujoco::mujoco" for configuration "Release"
set_property(TARGET mujoco::mujoco APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(mujoco::mujoco PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libmujoco.so.2.3.1"
  IMPORTED_SONAME_RELEASE "libmujoco.so.2.3.1"
  )

list(APPEND _IMPORT_CHECK_TARGETS mujoco::mujoco )
list(APPEND _IMPORT_CHECK_FILES_FOR_mujoco::mujoco "${_IMPORT_PREFIX}/lib/libmujoco.so.2.3.1" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
