@echo off
cd ../..

if not exist log\pointer_summarizer md log\pointer_summarizer

set logpath=%cd%\log\pointer_summarizer

cd models_repo\examples\text_summarization\pointer_summarizer\

set sed="C:\Program Files\Git\usr\bin\sed.exe"

%sed% -i "s/max_iterations = 100000/max_iterations = 30/g" config.py
%sed% -i "s/if iter % 5000 == 0 or iter == 1000:/if iter % 30 == 0 :/g" train.py

python train.py > %logpath%/train_%1.log 2>&1

if %ERRORLEVEL% == 1 (
    echo "exit_code: 1.0" >> %logpath%/train_%1.log
) else (
    echo "exit_code: 0.0" >> %logpath%/train_%1.log
)
