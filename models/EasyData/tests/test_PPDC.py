from easydata import EasyData
model = EasyData(model="image_orientation", device="cpu", return_res=True, print_res=False)
results = model.predict("./demo/image_orientation/")
print(results)
