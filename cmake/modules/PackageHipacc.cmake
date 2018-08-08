# Setup packaging variables
set(CPACK_PACKAGE_NAME Hipacc)
set(CPACK_PACKAGE_DESCRIPTION_SUMMARY "The Hipacc Framework: A Domain-Specific Language and Compiler for Image Processing")
set(CPACK_PACKAGE_VERSION_MAJOR ${HIPACC_MAJOR_VERSION})
set(CPACK_PACKAGE_VERSION_MINOR ${HIPACC_MINOR_VERSION})
set(CPACK_PACKAGE_VERSION_PATCH ${HIPACC_PATCH_VERSION})
set(CPACK_PACKAGE_VERSION ${HIPACC_VERSION})
set(CPACK_PACKAGE_CONTACT "https://hipacc-lang.org/#authors")
set(CPACK_RESOURCE_FILE_LICENSE ${CMAKE_CURRENT_SOURCE_DIR}/LICENSE)


# Set components to package
set(CPACK_COMPONENTS_GROUPING ALL_COMPONENTS_IN_ONE)
set(CPACK_COMPONENTS_ALL compiler runtime headers_runtime headers_dsl headers_clang libcxx tools samples)


# Setup Debian packaging options
if(UNIX AND NOT APPLE)
    set(CPACK_GENERATOR DEB)
    set(CPACK_DEBIAN_PACKAGE_SECTION "devel")
    set(CPACK_DEBIAN_PACKAGE_SUGGESTS "libopencv-dev (>= 2.4.0)")
    set(CPACK_DEBIAN_PACKAGE_HOMEPAGE "https://hipacc-lang.org")

    if(DEFINED PLATFORM)
        if(PLATFORM MATCHES 32)
            set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "i386")
        elseif(PLATFORM MATCHES 64)
            set(CPACK_DEBIAN_PACKAGE_ARCHITECTURE "amd64")
        endif(PLATFORM MATCHES 32)
    else(DEFINED PLATFORM)
        find_program(DPKG_PROGRAM dpkg DOC "dpkg program of Debian-based systems")
        if(DPKG_PROGRAM)
            execute_process(
                COMMAND ${DPKG_PROGRAM} --print-architecture
                OUTPUT_VARIABLE CPACK_DEBIAN_PACKAGE_ARCHITECTURE
                OUTPUT_STRIP_TRAILING_WHITESPACE)
        endif(DPKG_PROGRAM)
    endif(DEFINED PLATFORM)

    string(TOLOWER "${CPACK_PACKAGE_NAME}" CPACK_PACKAGE_NAME_LOWERCASE)

    set(CPACK_PACKAGE_FILE_NAME
        "${CPACK_PACKAGE_NAME_LOWERCASE}_${HIPACC_VERSION}_${CPACK_DEBIAN_PACKAGE_ARCHITECTURE}")

    if(NOT HIPACC_PACKAGING_PREFIX)
        set(HIPACC_PACKAGING_PREFIX "/usr/local")
    endif()
    set(CPACK_PACKAGING_INSTALL_PREFIX "${HIPACC_PACKAGING_PREFIX}/${CPACK_PACKAGE_NAME_LOWERCASE}-${HIPACC_VERSION}")
    set(HIPACC_PACKAGE_SYMLINK "${HIPACC_PACKAGING_PREFIX}/${CPACK_PACKAGE_NAME_LOWERCASE}")

    # postinst script for creating symlink and profile file
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/postinst "\
ln -s ${CPACK_PACKAGE_NAME_LOWERCASE}-${HIPACC_VERSION} ${HIPACC_PACKAGE_SYMLINK}
echo \"export PATH=${HIPACC_PACKAGE_SYMLINK}/bin:\\\$PATH\" > /etc/profile.d/${CPACK_PACKAGE_NAME_LOWERCASE}.sh
echo \"export CPLUS_INCLUDE_PATH=${HIPACC_PACKAGE_SYMLINK}/include:\\\$CPLUS_INCLUDE_PATH\" >> /etc/profile.d/${CPACK_PACKAGE_NAME_LOWERCASE}.sh
echo \"export LIBRARY_PATH=${HIPACC_PACKAGE_SYMLINK}/lib:\\\$LIBRARY_PATH\" >> /etc/profile.d/${CPACK_PACKAGE_NAME_LOWERCASE}.sh")

    # postrm script for cleaning symlink and profile file
    file(WRITE ${CMAKE_CURRENT_BINARY_DIR}/postrm "\
rm -f ${HIPACC_PACKAGE_SYMLINK}
rm -f /etc/profile.d/${CPACK_PACKAGE_NAME_LOWERCASE}.sh")

    # add extra scripts
    set(CPACK_DEBIAN_PACKAGE_CONTROL_EXTRA
        ${CMAKE_CURRENT_BINARY_DIR}/postinst;
        ${CMAKE_CURRENT_BINARY_DIR}/postrm)

    # enable component install for Debian
    set(CPACK_DEB_COMPONENT_INSTALL ON)

    # Strip development machine's path from compiler and runtime includes
    set(CU_COMPILER "nvcc")
    set(CL_COMPILER "${CPACK_PACKAGING_INSTALL_PREFIX}/bin/cl_compile")
    set(RUNTIME_INCLUDES "${CPACK_PACKAGING_INSTALL_PREFIX}/include")
endif(UNIX AND NOT APPLE)


# Setup macOS app bundle
if(APPLE)
    set(CPACK_GENERATOR DragNDrop)
    set(MACOSX_BUNDLE_SHORT_VERSION_STRING ${HIPACC_VERSION})
    set(MACOSX_BUNDLE_LONG_VERSION_STRING ${HIPACC_VERSION})
    set(MACOSX_BUNDLE_INFO_STRING "${CPACK_PACKAGE_NAME} ${HIPACC_VERSION}")

    set(MACOSX_BUNDLE_ICON_FILE hipacc.icns)
    set_source_files_properties(${CMAKE_SOURCE_DIR}/res/hipacc.icns
        PROPERTIES MACOSX_PACKAGE_LOCATION Resources)

    include(BundleUtilities)
    fixup_bundle("${CMAKE_BINARY_DIR}/${CPACK_PACKAGE_NAME}-${HIPACC_VERSION}.app" "${MAXOSX_LIBRARY_DIRS}" "")
endif(APPLE)


# Setup Windows Nullsoft Scriptable Install System
if(WIN32)
    set(CPACK_GENERATOR NSIS)
    set(CPACK_PACKAGE_INSTALL_DIRECTORY
        ${CPACK_PACKAGE_NAME}-${HIPACC_VERSION})
    list(APPEND CMAKE_MODULE_PATH ${CMAKE_SOURCE_DIR}/cmake/templates)

    # setup icons and banner
    set(CPACK_NSIS_MUI_ICON ${CMAKE_SOURCE_DIR}/res/hipacc.ico)
    set(CPACK_NSIS_MUI_UNIICON ${CMAKE_SOURCE_DIR}/res/hipacc.ico)
    set(CPACK_NSIS_MUI_HEADERIMAGE ${CMAKE_SOURCE_DIR}/res/hipacc_banner.bmp)
    string(REPLACE "/" "\\\\"
           CPACK_NSIS_MUI_HEADERIMAGE "${CPACK_NSIS_MUI_HEADERIMAGE}")

    # find and add Visual Studio redistributables
    include(InstallRequiredSystemLibraries)
    set(MSVC_REDIST_NAME "vcredist_${CMAKE_MSVC_ARCH}.exe")
    find_program(MSVC_REDIST NAMES ${MSVC_REDIST_NAME}
                 PATHS "${MSVC_REDIST_DIR}" "${MSVC_REDIST_DIR}/*")
    if(MSVC_REDIST)
        install(FILES ${MSVC_REDIST} DESTINATION . COMPONENT vcredist)
        list(APPEND CPACK_COMPONENTS_ALL vcredist)
        set(CMAKE_NSIS_MSVC_REDIST_INSTALL
            "ExecWait \\\"\$INSTDIR\\\\${MSVC_REDIST_NAME} /passive\\\"")
    else(MSVC_REDIST)
        message(WARNING "Could not find Visual Studio redistributables!")
    endif(MSVC_REDIST)

    # set HIPACC_PATH and append to PATH environment variables
    set(CPACK_NSIS_EXTRA_INSTALL_COMMANDS "\
StrCpy \$ADD_TO_PATH_ALL_USERS \\\"1\\\"
Push \\\"HIPACC_PATH\\\"
Push \\\"\$INSTDIR\\\"
Call AddToEnvVar
Push \\\"\$INSTDIR/bin\\\"
Call AddToPath
${CMAKE_NSIS_MSVC_REDIST_INSTALL}")

    # unset HIPACC_PATH and remove from PATH environment variables
    set(CPACK_NSIS_EXTRA_UNINSTALL_COMMANDS "\
StrCpy \$ADD_TO_PATH_ALL_USERS \\\"1\\\"
Push \\\"HIPACC_PATH\\\"
Push \\\"\$INSTDIR\\\"
Call un.RemoveFromEnvVar
Push \\\"\$INSTDIR/bin\\\"
Call un.RemoveFromPath")

    # Strip development machine's path from compiler and runtime includes
    set(CU_COMPILER "nvcc.exe")
    set(CL_COMPILER "cl_compile.exe")
    set(RUNTIME_INCLUDES "%HIPACC_PATH%/include")
endif(WIN32)


include(CPack)
