@echo off

set cur_path=%cd%
echo "++++++++++++++++++++++++++++++++%1 begin to install !!!!!!!!!++++++++++++++++++++++++++++++++"

set log_path=%cur_path%

del /q %1_install.log

setlocal enabledelayedexpansion
hub install %1 >> %log_path%/%1_install.log 2>&1
if not !errorlevel! == 0 (
    echo "exit_code: 1.0" >> %log_path%/EXIT_%1_install.log
) else (
    echo "exit_code: 0.0" >> %log_path%/EXIT_%1_install.log
)
