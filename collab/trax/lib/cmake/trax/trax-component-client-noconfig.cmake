#----------------------------------------------------------------
# Generated CMake target import file.
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "trax_client" for configuration ""
set_property(TARGET trax_client APPEND PROPERTY IMPORTED_CONFIGURATIONS NOCONFIG)
set_target_properties(trax_client PROPERTIES
  IMPORTED_LOCATION_NOCONFIG "${_IMPORT_PREFIX}/lib/libtrax_client.so"
  IMPORTED_NO_SONAME_NOCONFIG "TRUE"
  )

list(APPEND _IMPORT_CHECK_TARGETS trax_client )
list(APPEND _IMPORT_CHECK_FILES_FOR_trax_client "${_IMPORT_PREFIX}/lib/libtrax_client.so" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
