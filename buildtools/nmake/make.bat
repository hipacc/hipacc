@if "%VS140COMNTOOLS%"=="" goto error_vc14
@call "%VS140COMNTOOLS%\..\..\VC\vcvarsall.bat" x64
@if "%HIPACC_PATH%"=="" goto error_hipacc
@if "%OPENCV_DIR%"=="" goto target_rules


:check_opencv3
@where.exe /R %OPENCV_DIR%\lib opencv_world*.lib > worldlib.txt
@set /p OPENCV_LIBARY_WORLD=<worldlib.txt
@del worldlib.txt
@if "%OPENCV_LIBARY_WORLD%"=="" goto check_opencv2
@set CXX_FLAGS_OPENCV=/D USE_OPENCV /I "%OPENCV_DIR%\..\..\include" "%OPENCV_LIBARY_WORLD%"
@set NVCC_FLAGS_OPENCV=-D USE_OPENCV -I "%OPENCV_DIR%/../../include" -l "%OPENCV_LIBARY_WORLD:~0,-4%"
@goto target_rules

:check_opencv2
@where.exe /R %OPENCV_DIR%\lib opencv_core*.lib > corelib.txt
@set /p OPENCV_LIBARY_CORE=<corelib.txt
@del corelib.txt
@if "%OPENCV_LIBARY_CORE%"=="" goto target_rules
@where.exe /R %OPENCV_DIR%\lib opencv_highgui*.lib > highguilib.txt
@set /p OPENCV_LIBARY_HIGHGUI=<highguilib.txt
@del highguilib.txt
@if "%OPENCV_LIBARY_HIGHGUI%"=="" goto target_rules
@where.exe /R %OPENCV_DIR%\lib opencv_imgproc*.lib > imgproclib.txt
@set /p OPENCV_LIBARY_IMGPROC=<imgproclib.txt
@del imgproclib.txt
@if "%OPENCV_LIBARY_IMGPROC%"=="" goto target_rules
@set CXX_FLAGS_OPENCV=/D USE_OPENCV /I "%OPENCV_DIR%\..\..\include" "%OPENCV_LIBARY_CORE%" "%OPENCV_LIBARY_HIGHGUI%" "%OPENCV_LIBARY_IMGPROC%"
@set NVCC_FLAGS_OPENCV=-D USE_OPENCV -I "%OPENCV_DIR%/../../include" -l "%OPENCV_LIBARY_CORE:~0,-4%" -l "%OPENCV_LIBARY_HIGHGUI:~0,-4%" -l "%OPENCV_LIBARY_IMGPROC:~0,-4%"
@goto target_rules


:target_rules
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
