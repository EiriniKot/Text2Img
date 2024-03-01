import os
from tools import TxtImage
import streamlit as st

QDRANT_HOST = os.environ['QDRANT_HOST']
QDRANT_PORT = os.environ['QDRANT_PORT']

# Initialize your ImgToTextGenerator and TxtImage objects
txt_img = TxtImage(QDRANT_HOST, QDRANT_PORT, "ecommerce_collection")


def main():
    st.title('Text to Image Translator')
    # Get user input text
    user_text = st.text_input("Enter a text:")

    if st.button("Nearest Image"):
        print("button pressed")
        # Search for images based on the input text
        image_paths = txt_img.search_image(user_text, limit=1, score_threshold=0.7)
        if len(image_paths) > 0:
            # Display images
            for img_path in image_paths:
                st.image(img_path.payload["name"])
        else:
            st.text("No similar image found")


if __name__ == "__main__":
    main()
