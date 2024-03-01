from tools import ImgToTextGenerator

image_paths = ["dataset/94880.jpg"]
text_gen = ImgToTextGenerator(model_path="/home/eirini/blip-image-captioning-large")
out = text_gen.predict_step(image_paths, batch_size=100)
print(out[0])
#https://www.kaggle.com/datasets/shamsaddin97/image-captioning-dataset-random-images?resource=download