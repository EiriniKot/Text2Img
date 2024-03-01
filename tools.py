from qdrant_client.http.models import Distance, VectorParams
from qdrant_client import QdrantClient, models

from transformers import VisionEncoderDecoderModel, ViTImageProcessor, AutoTokenizer
from sentence_transformers import SentenceTransformer
import torch
from PIL import Image
from tqdm import tqdm
from time import sleep


class TxtImage:
    def __init__(self, location, port, collection_name):
        self.client = QdrantClient(host=location, port=port)
        self.encoder = SentenceTransformer("all-MiniLM-L6-v2")


        self.collection_name = collection_name
        collections_names = [c.name for c in self.client.get_collections().collections]
        if self.collection_name not in collections_names:
            self.client.recreate_collection(collection_name=self.collection_name,
                                            vectors_config=VectorParams(size=self.encoder.get_sentence_embedding_dimension(),
                                                                      distance=Distance.DOT))

    def upload_images_description(self, captions, batch_size = 100):
        """
        captions = [{"name": "90.jpg", "description": "A man travels through time."},...,]
        :param captions: list of captions
        :param batch_size: int
        :return:
        """
        points = [models.PointStruct(id=idx, vector=self.encoder.encode(cap["description"]).tolist(), payload=cap) for idx,
        cap in enumerate(captions)]
        num_points = len(points)

        num_batches = (num_points + batch_size - 1) // batch_size  # Calculate
        for batch_idx in tqdm(range(num_batches)):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, num_points)
            batch_points = points[start_idx:end_idx]
            self.client.upsert(collection_name=self.collection_name, points=batch_points)

    def search_image(self, query, limit=3, score_threshold=0.5):
        emb = self.encoder.encode(query)
        print(emb.shape)
        search_result = self.client.search(collection_name=self.collection_name, query_vector=emb, limit=limit,
                                           score_threshold=score_threshold)
        print('search_result', search_result)

        return search_result

    def close(self):
        self.client.close()


class ImgToTextGenerator:
    def __init__(self, model_path):

        self.model = VisionEncoderDecoderModel.from_pretrained(model_path)
        self.feature_extractor = ViTImageProcessor.from_pretrained(model_path)
        self.tokenizer = AutoTokenizer.from_pretrained(model_path)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

        max_length = 12
        num_beams = 4
        self.gen_kwargs = {"max_length": max_length, "num_beams": num_beams}
        print('Generator initialized')

    def predict_step(self, image_paths, batch_size=100):
        len_paths = len(image_paths)
        dict_captions = []
        num_batches = (len_paths + batch_size - 1) // batch_size  # Calculate
        for batch_idx in tqdm(range(num_batches)):
            start_idx = batch_idx * batch_size
            end_idx = min((batch_idx + 1) * batch_size, len_paths)

            batch_image_paths = image_paths[start_idx:end_idx]
            batch_images = []

            for path in batch_image_paths:
                i_image = Image.open(path)
                if i_image.mode != "RGB":
                    i_image = i_image.convert(mode="RGB")
                batch_images.append(i_image)

            pixel_values = self.feature_extractor(images=batch_images, return_tensors="pt").pixel_values
            pixel_values = pixel_values.to(self.device)

            output_ids = self.model.generate(pixel_values, **self.gen_kwargs)

            preds = self.tokenizer.batch_decode(output_ids, skip_special_tokens=True)
            preds = [pred.strip() for pred in preds]
            outs = list(zip(image_paths, preds))
            dict_captions.extend(self.generate_dict_of_captions(outs))
        return dict_captions

    @staticmethod
    def generate_dict_of_captions(list_of_captions):
        dictionary_of_captions = [{"name": path, "description": caption} for path, caption in list_of_captions]
        return dictionary_of_captions
