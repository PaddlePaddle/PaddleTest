pip install tf-models-official
pip install tf-models-nightly
git clone https://github.com/tensorflow/models.git
export PYTHONPATH=$PYTHONPATH:`pwd`/models
cd models
pip install --user -r official/requirements.txt
