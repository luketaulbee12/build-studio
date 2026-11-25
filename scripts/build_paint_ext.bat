@echo off
call "C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat" >nul
set DISTUTILS_USE_SDK=1
set TORCH_CUDA_ARCH_LIST=8.9
set CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.9
set PY=C:\Users\Luke\Desktop\Projects\recon\.venv\Scripts\python.exe
cd /d C:\Users\Luke\Desktop\Projects\recon\Hunyuan3D-2\hy3dgen\texgen\differentiable_renderer
%PY% setup.py build_ext --inplace || exit /b 1
cd /d C:\Users\Luke\Desktop\Projects\recon\Hunyuan3D-2\hy3dgen\texgen\custom_rasterizer
%PY% -m pip install --no-build-isolation . || exit /b 1
echo BUILD OK
