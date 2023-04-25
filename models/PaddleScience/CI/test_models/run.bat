@echo off

set cases=

for /r %%i in (test_*.py) do set cases=!cases! "%%~i"
set serial_bug=0
set distributed_bug=0
set bug=0

set CUDA_VISIBLE_DEVICES=%cudaid2%
echo "===== examples bug list =====" > result.txt
echo "serial bug list:" >> result.txt
for %%i in (%cases%) do (
    echo serial %%i test
    python %py_version% %%i
    if "!errorlevel!" neq "0" (
        echo %%i >> result.txt
        set /a bug+=1
        set /a serial_bug+=1
    )
)
echo "serial bugs: %serial_bug%" >> result.txt

rem set CUDA_VISIBLE_DEVICES=0,1
rem echo "distributed bug list:" >> result.txt
rem for %%i in (%cases%) do (
rem     echo distributed %%i test
rem     python3.7 -m paddle.distributed.launch --devices=0,1  %%i
rem     if "!errorlevel!" neq "0" (
rem         echo %%i >> result.txt
rem         set /a bug+=1
rem         set /a distributed_bug+=1
rem     )
rem )
rem echo "distributed bugs: %distributed_bug%" >> result.txt

echo "total bugs: %bug%" >> result.txt
type result.txt
exit %bug%
