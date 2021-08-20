@echo off
cd ../..

cd models_repo\examples\text_summarization\pointer_summarizer\

xcopy /y /c /h /r D:\ce_data\paddleNLP\pointer_summarizer\*  .\
