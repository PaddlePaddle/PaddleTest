import os
import yaml
path = "/workspace/PaddleTest_wht/models_restruct/PaddleSlim/cases"
files = os.listdir(path)

file_data = ""
for file in files:
	with open(path +"/"+ file, "r", encoding="utf-8") as f:
		content = yaml.load(f, Loader=yaml.FullLoader)
		if content["case"].get("mac"):
			print(file)