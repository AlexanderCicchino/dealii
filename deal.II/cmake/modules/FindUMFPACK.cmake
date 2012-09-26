#
# Try to find the UMFPACK library
#

INCLUDE(FindPackageHandleStandardArgs)

SET_IF_EMPTY(AMD_DIR "$ENV{AMD_DIR}")
SET_IF_EMPTY(UMFPACK_DIR "$ENV{UMFPACK_DIR}")

FIND_PATH(UMFPACK_INCLUDE_DIR umfpack.h
  HINTS
    ${AMD_DIR}
    ${UMFPACK_DIR}
  PATH_SUFFIXES
    umfpack include/umfpack include Include UMFPACK/Include ../UMFPACK/Include
)

FIND_LIBRARY(UMFPACK_LIBRARY
  NAMES umfpack
  HINTS
    ${UMFPACK_DIR}
  PATH_SUFFIXES
    lib${LIB_SUFFIX} lib64 lib Lib UMFPACK/Lib ../UMFPACK/Lib
)

FIND_PACKAGE_HANDLE_STANDARD_ARGS(UMFPACK DEFAULT_MSG UMFPACK_LIBRARY UMFPACK_INCLUDE_DIR)

IF(UMFPACK_FOUND)
  MARK_AS_ADVANCED(
    UMFPACK_LIBRARY
    UMFPACK_INCLUDE_DIR
    UMFPACK_DIR
  )
ELSE()
  SET(UMFPACK_DIR "" CACHE STRING
    "An optional hint to an UMFPACK directory"
    )
ENDIF()

