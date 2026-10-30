# SPDX-FileCopyrightText: Copyright 2023 Francis Williams
# SPDX-License-Identifier: MIT
#
# Unmodified source from point-cloud-utils at the MIT-licensed snapshot:
# https://github.com/fwilliams/point-cloud-utils/blob/0dd1bf4352a01a7297479e5a7c9ffc98e132ba92/cmake/FindMKL.cmake
#
# Copyright 2023 Francis Williams
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

if (MKL_LIBRARIES)
  set(MKL_FIND_QUIETLY TRUE)
endif (MKL_LIBRARIES)

# 64-bit system
if(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "x86_64")
  find_library(MKL_LIBRARIES
    mkl_core
    PATHS
    $ENV{MKLLIB}
    /opt/intel/mkl/*/lib/em64t
    /opt/intel/Compiler/*/*/mkl/lib/em64t
    ${LIB_INSTALL_DIR}
  )

  find_library(MKL_GUIDE
    guide
    PATHS
    $ENV{MKLLIB}
    /opt/intel/mkl/*/lib/em64t
    /opt/intel/Compiler/*/*/mkl/lib/em64t
    /opt/intel/Compiler/*/*/lib/intel64
    ${LIB_INSTALL_DIR}
  )

  find_path(MKL_INCLUDE_DIR
    mkl.h
  )

  if(MKL_LIBRARIES AND MKL_GUIDE)
    set(MKL_LIBRARIES ${MKL_LIBRARIES} mkl_intel_lp64 mkl_sequential ${MKL_GUIDE} pthread)
    set(MKL_INCLUDE_DIRS ${MKL_INCLUDE_DIR})
  endif()

# 32-bit system
else(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "x86_64")
  find_library(MKL_LIBRARIES
    mkl_core
    PATHS
    $ENV{MKLLIB}
    /opt/intel/mkl/*/lib/32
    /opt/intel/Compiler/*/*/mkl/lib/32
    ${LIB_INSTALL_DIR}
  )

  find_library(MKL_GUIDE
    guide
    PATHS
    $ENV{MKLLIB}
    /opt/intel/mkl/*/lib/32
    /opt/intel/Compiler/*/*/mkl/lib/32
    /opt/intel/Compiler/*/*/lib/intel32
    ${LIB_INSTALL_DIR}
  )

  find_path(MKL_INCLUDE_DIR
    mkl.h
  )

  if(MKL_LIBRARIES AND MKL_GUIDE)
    set(MKL_LIBRARIES ${MKL_LIBRARIES} mkl_intel mkl_sequential ${MKL_GUIDE} pthread)
  endif()

endif(${CMAKE_HOST_SYSTEM_PROCESSOR} STREQUAL "x86_64")

include(FindPackageHandleStandardArgs)
find_package_handle_standard_args(MKL DEFAULT_MSG MKL_LIBRARIES)

if (${MKL_FOUND})
  set(MKL_LIBRARIES ${MKL_LIBRARIES})
  set(MKL_INCLUDE_DIRS ${MKL_INCLUDE_DIR})
endif(${MKL_FOUND})

mark_as_advanced(MKL_LIBRARIES)
