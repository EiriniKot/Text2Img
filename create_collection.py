import os
from tools import ImgToTextGenerator, TxtImage
import pickle

QDRANT_HOST = os.environ.get('QDRANT_HOST', 'localhost')
QDRANT_PORT = os.environ.get('QDRANT_PORT', '6333')
MODEL_PATH = os.environ.get('MODEL_PATH', 'nlpconnect/vit-gpt2-image-captioning')

coll_name = "ecommerce_collection"
coll_filepath = "collection.pkl"

if not os.path.isfile(coll_filepath):
    # Initialize your ImgToTextGenerator and TxtImage objects
    text_gen = ImgToTextGenerator(model_path=MODEL_PATH)
    image_paths_dt1 = [os.path.join("dataset/1", filenm) for filenm in os.listdir("dataset/1")]
    image_paths_dt2 = [os.path.join("dataset/2", filenm) for filenm in os.listdir("dataset/2")]
    image_paths = image_paths_dt1 + image_paths_dt2
    out = text_gen.predict_step(image_paths, batch_size=100)

    # Save to a pickle file
    with open(coll_filepath, "wb") as pickle_file:
        pickle.dump(out, pickle_file)


#%%%
# First we need to pull image and run the container
# sudo docker pull qdrant/qdrant
# sudo docker run -p 6333:6333 -p 6335:6335     -v $(pwd)/qdrant_storage:/qdrant/storage:z     qdrant/qdrant

# Load data from pickle file
with open(coll_filepath, "rb") as pickle_file:
    collection_data = pickle.load(pickle_file)

txt_img = TxtImage(QDRANT_HOST, QDRANT_PORT, coll_name)
txt_img.upload_images_description(collection_data, batch_size=100)
