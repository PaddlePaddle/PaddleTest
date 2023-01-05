### Imagen text encoder preparing
Imagen need load pretrained text encoder model for the training loop. T5 and
DeBERTa V2 are provided for Imagen.
#### T5-11B
``` 
# T5 tokenizer and model was converted from Huggingface.
config.json: wget https://paddlefleetx.bj.bcebos.com/tokenizers/t5/t5-11b/config.json
spiece.model: wget https://paddlefleetx.bj.bcebos.com/tokenizers/t5/t5-11b/spiece.model
tokenizer.json: wget https://paddlefleetx.bj.bcebos.com/tokenizers/t5/t5-11b/tokenizer.json
t5 model: wget https://fleetx.bj.bcebos.com/T5/t5-11b/t5.pd.tar.gz.0
          wget https://fleetx.bj.bcebos.com/T5/t5-11b/t5.pd.tar.gz.1
          wget https://fleetx.bj.bcebos.com/T5/t5-11b/t5.pd.tar.gz.2
          wget https://fleetx.bj.bcebos.com/T5/t5-11b/t5.pd.tar.gz.3
          wget https://fleetx.bj.bcebos.com/T5/t5-11b/t5.pd.tar.gz.4
          cat t5.pd.tar.gz.* |tar -xf - 
then put them into t5 folder like this:
projects/imagen/t5
                 ├── t5-11b
                    ├── config.json
                    ├── spiece.model
                    ├── t5.pd
                    └── tokenizer.json
``` 

#### DeBERTa V2 1.5B
```
# DeBERTa V2 tokenizer and model was converted from Huggingface.
config.json: wget https://paddlefleetx.bj.bcebos.com/tokenizers/debertav2/config.json
spm.model: wget https://paddlefleetx.bj.bcebos.com/tokenizers/debertav2/spm.model
tokenizer_config.json: https://paddlefleetx.bj.bcebos.com/tokenizers/debertav2/tokenizer_config.json
denerta v2 model: wget https://fleetx.bj.bcebos.com/DebertaV2/debertav2.pd.tar.gz.0
                  wget https://fleetx.bj.bcebos.com/DebertaV2/debertav2.pd.tar.gz.1
                  tar debertav2.pd.tar.gz.* | tar -xf -
then put them into cache folder like this:
projects/imagen/cache
                  └── deberta-v-xxlarge
                      ├── config.json
                      ├── debertav2.pd
                      ├── spm.model
                      ├── tokenizer_config.json
                  tar debertav2.pd.tar.gz.* | tar -xf -
then put them into cache folder like this:
projects/imagen/cache
                  └── deberta-v-xxlarge
                      ├── config.json
                      ├── debertav2.pd
                      ├── spm.model
                      ├── tokenizer_config.json