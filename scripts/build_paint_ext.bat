@echo off
rem Builds Hunyuan3D-2 Paint's two native add-ons (needs VS 2022 Build Tools + CUDA 12.x).
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set DISTUTILS_USE_SDK=1
if "%TORCH_CUDA_ARCH_LIST%"=="" set TORCH_CUDA_ARCH_LIST=8.9
set ROOT=%~dp0..
set PY=%ROOT%\.venv\Scripts\python.exe
cd /d "%ROOT%\Hunyuan3D-2\hy3dgen\texgen\differentiable_renderer"
"%PY%" setup.py build_ext --inplace || exit /b 1
cd /d "%ROOT%\Hunyuan3D-2\hy3dgen\texgen\custom_rasterizer"
"%PY%" -m pip install --no-build-isolation . || exit /b 1
echo BUILD OK
