@echo off
setlocal
python -X utf8 -m nonlinearSolver.native.build %*
exit /b %errorlevel%
