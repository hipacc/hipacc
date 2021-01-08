# ./cmake/modules/VersionHipacc.cmake

macro(get_hipacc_version)
  find_package(Git)
  if(Git_FOUND)
      execute_process(COMMAND ${GIT_EXECUTABLE} describe --tags --always
                      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                      OUTPUT_VARIABLE HIPACC_VERSION)
      string(STRIP "${HIPACC_VERSION}" HIPACC_VERSION)
  endif()

  if(NOT HIPACC_VERSION)
      message(FATAL_ERROR "Could not determine version of Hipacc!")
  endif()

  string(REGEX REPLACE "^v(.*)" "\\1" HIPACC_VERSION "${HIPACC_VERSION}")
  string(REGEX REPLACE "^([0-9]+)\\..*" "\\1" HIPACC_VERSION_MAJOR "${HIPACC_VERSION}")
  string(REGEX REPLACE "^[0-9]+\\.([0-9]+).*" "\\1" HIPACC_VERSION_MINOR "${HIPACC_VERSION}")
  string(REGEX REPLACE "^[0-9]+\\.[0-9]+\\.([0-9]+).*" "\\1" HIPACC_VERSION_PATCH "${HIPACC_VERSION}")
  string(REGEX REPLACE "^[0-9]+\\.[0-9]+\\.[0-9]+(.*)" "\\1" HIPACC_VERSION_TWEAK "${HIPACC_VERSION}")

  # get git repository and revision
  if(EXISTS ${CMAKE_SOURCE_DIR}/.git)

      execute_process(COMMAND git remote get-url origin
                      WORKING_DIRECTORY ${CMAKE_SOURCE_DIR}
                      TIMEOUT 5
                      RESULT_VARIABLE git_result
                      OUTPUT_VARIABLE HIPACC_GIT_REPOSITORY
                      ERROR_QUIET)

      if(HIPACC_GIT_REPOSITORY)
          string(STRIP ${HIPACC_GIT_REPOSITORY} HIPACC_GIT_REPOSITORY)
          string(REGEX REPLACE "://.+@" "://" HIPACC_GIT_REPOSITORY ${HIPACC_GIT_REPOSITORY})
      endif()

  endif()

  if(NOT HIPACC_GIT_REPOSITORY)
      set(HIPACC_GIT_REPOSITORY "https://github.com/hipacc/hipacc/releases")
  endif()

  set(HIPACC_GIT_VERSION "${HIPACC_VERSION}")
endmacro()

