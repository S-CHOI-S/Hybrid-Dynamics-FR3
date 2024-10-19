#----------------------------------------------------------------
# Generated CMake target import file for configuration "Release".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "Qhull::qhull" for configuration "Release"
set_property(TARGET Qhull::qhull APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(Qhull::qhull PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/qhull"
  )

list(APPEND _IMPORT_CHECK_TARGETS Qhull::qhull )
list(APPEND _IMPORT_CHECK_FILES_FOR_Qhull::qhull "${_IMPORT_PREFIX}/bin/qhull" )

# Import target "Qhull::rbox" for configuration "Release"
set_property(TARGET Qhull::rbox APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(Qhull::rbox PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/rbox"
  )

list(APPEND _IMPORT_CHECK_TARGETS Qhull::rbox )
list(APPEND _IMPORT_CHECK_FILES_FOR_Qhull::rbox "${_IMPORT_PREFIX}/bin/rbox" )

# Import target "Qhull::qconvex" for configuration "Release"
set_property(TARGET Qhull::qconvex APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(Qhull::qconvex PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/qconvex"
  )

list(APPEND _IMPORT_CHECK_TARGETS Qhull::qconvex )
list(APPEND _IMPORT_CHECK_FILES_FOR_Qhull::qconvex "${_IMPORT_PREFIX}/bin/qconvex" )

# Import target "Qhull::qdelaunay" for configuration "Release"
set_property(TARGET Qhull::qdelaunay APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(Qhull::qdelaunay PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/qdelaunay"
  )

list(APPEND _IMPORT_CHECK_TARGETS Qhull::qdelaunay )
list(APPEND _IMPORT_CHECK_FILES_FOR_Qhull::qdelaunay "${_IMPORT_PREFIX}/bin/qdelaunay" )

# Import target "Qhull::qvoronoi" for configuration "Release"
set_property(TARGET Qhull::qvoronoi APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(Qhull::qvoronoi PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/qvoronoi"
  )

list(APPEND _IMPORT_CHECK_TARGETS Qhull::qvoronoi )
list(APPEND _IMPORT_CHECK_FILES_FOR_Qhull::qvoronoi "${_IMPORT_PREFIX}/bin/qvoronoi" )

# Import target "Qhull::qhalf" for configuration "Release"
set_property(TARGET Qhull::qhalf APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(Qhull::qhalf PROPERTIES
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/bin/qhalf"
  )

list(APPEND _IMPORT_CHECK_TARGETS Qhull::qhalf )
list(APPEND _IMPORT_CHECK_FILES_FOR_Qhull::qhalf "${_IMPORT_PREFIX}/bin/qhalf" )

# Import target "Qhull::qhullcpp" for configuration "Release"
set_property(TARGET Qhull::qhullcpp APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(Qhull::qhullcpp PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "CXX"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libqhullcpp.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS Qhull::qhullcpp )
list(APPEND _IMPORT_CHECK_FILES_FOR_Qhull::qhullcpp "${_IMPORT_PREFIX}/lib/libqhullcpp.a" )

# Import target "Qhull::qhullstatic" for configuration "Release"
set_property(TARGET Qhull::qhullstatic APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(Qhull::qhullstatic PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libqhullstatic.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS Qhull::qhullstatic )
list(APPEND _IMPORT_CHECK_FILES_FOR_Qhull::qhullstatic "${_IMPORT_PREFIX}/lib/libqhullstatic.a" )

# Import target "Qhull::qhullstatic_r" for configuration "Release"
set_property(TARGET Qhull::qhullstatic_r APPEND PROPERTY IMPORTED_CONFIGURATIONS RELEASE)
set_target_properties(Qhull::qhullstatic_r PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_RELEASE "C"
  IMPORTED_LOCATION_RELEASE "${_IMPORT_PREFIX}/lib/libqhullstatic_r.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS Qhull::qhullstatic_r )
list(APPEND _IMPORT_CHECK_FILES_FOR_Qhull::qhullstatic_r "${_IMPORT_PREFIX}/lib/libqhullstatic_r.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
