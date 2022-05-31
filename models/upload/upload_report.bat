@echo off
set compile_path=%1
set build_repo_name=%2
set build_repo_branch=%3
set build_url=%4
set build_exit_code=%5
set build_status=%6

for /f "tokens=1,2,3,4,5 delims=/" %%a in ( "%compile_path%" ) do (
set a=%%a
set b=%%b
set c=%%c
set d=%%d
set e=%%e
)
set path_temp=%a%/%b%/%c%/%d%/%e%
wget %path_temp%/description.txt --no-check-certificate

for /f "delims=" %%t in ('findstr commit_id description.txt ') do set str1=%%t
for /f "tokens=1,2,3,4,5 delims=:" %%a in ( "%str1%" ) do set build_commit_id=%%b

echo %build_commit_id%

for /f "delims=" %%t in ('findstr commit_time description.txt ') do set str1=%%t
for /f "tokens=1,2,3,4,5 delims=:" %%a in ( "%str1%" ) do set build_commit_time=%%b

echo %build_commit_time%

set build_type_id=%AGILE_PIPELINE_CONF_ID%
set build_id=%AGILE_PIPELINE_BUILD_ID%
set build_job_id=%AGILE_JOB_BUILD_ID%

set http_proxy=
set https_proxy=
python upload_report.py
