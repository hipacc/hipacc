@if "%VS140COMNTOOLS%"=="" goto error_vc14
@call "%VS140COMNTOOLS%\..\..\VC\vcvarsall.bat"
@if "%HIPACC_PATH%"=="" goto error_hipacc


@if "%1" == "cpu" goto cpu
@if "%1" == "cuda" goto cuda
@if "%1" == "clean" goto clean
@if "%1" == "distclean" goto distclean


:cpu
nmake.exe /f sample.mak cpu
@goto end

:cuda
@if "%CUDA_PATH%"=="" goto error_cuda
nmake.exe /f sample.mak cuda
@goto end

:clean
nmake.exe /f sample.mak clean
@goto end

:distclean
nmake.exe /f sample.mak distclean
@goto end


:error_cuda
@echo ERROR: Cannot find CUDA installation.
@goto end

:error_hipacc
@echo ERROR: Cannot find Hipacc installation.
@goto end

:error_vc14
@echo ERROR: Cannot find Visual C++ Build Tools 2015.
@goto end

:end