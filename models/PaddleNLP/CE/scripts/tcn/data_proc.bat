@echo off
cd ../..
cd models_repo\examples\time_series\tcn
del time_series_covid19_confirmed_global.csv
python -m wget https://raw.githubusercontent.com/CSSEGISandData/COVID-19/master/csse_covid_19_data/csse_covid_19_time_series/time_series_covid19_confirmed_global.csv
