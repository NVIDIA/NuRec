# SPDX-FileCopyrightText: Copyright 2023 Francis Williams
# SPDX-License-Identifier: MIT
#
# Unmodified source from point-cloud-utils at the MIT-licensed snapshot:
# https://github.com/fwilliams/point-cloud-utils/blob/0dd1bf4352a01a7297479e5a7c9ffc98e132ba92/cmake/DownloadExternalDeps.cmake
#
# Copyright 2023 Francis Williams
#
# Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the “Software”), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED “AS IS”, WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.

include(DownloadProject)

# With CMake 3.8 and above, we can hide warnings about git being in a
# detached head by passing an extra GIT_CONFIG option
if(NOT (${CMAKE_VERSION} VERSION_LESS "3.8.0"))
  set(LIBIGL_EXTRA_OPTIONS "GIT_CONFIG advice.detachedHead=false")
else()
  set(LIBIGL_EXTRA_OPTIONS "")
endif()

# Shortcut function
function(download_dep name)
  download_project(
    PROJ         ${name}
    SOURCE_DIR   ${EXTERNAL_DEP_DIR}/${name}
    DOWNLOAD_DIR ${EXTERNAL_DEP_DIR}/.cache/${name}
    QUIET
    ${ARGN}
  )
endfunction()
