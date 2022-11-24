from easydata import EasyData

ppdataaug = EasyData(
    model="ppdataaug",
    ori_data_dir="demo/clas_data",
    label_file="demo/clas_data/train_list.txt",
    gen_mode="img2img",
    model_type="cls",
)
ppdataaug.predict()
