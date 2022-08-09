@echo off
cd ../..

cd models_repo\examples\text_graph\erniesage\

set sed="C:\Program Files\Git\usr\bin\sed.exe"

%sed% -i "s/batch_size: 32/batch_size: 8/g" ./config/erniesage_link_prediction.yaml

%sed% -i "s/epoch: 30/epoch: 1/g" ./config/erniesage_link_prediction.yaml

md graph_workdir

python ./preprocessing/dump_graph.py --conf ./config/erniesage_link_prediction.yaml
