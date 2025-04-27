# local_image_search_app.py

import os
import torch
import faiss
import numpy as np
import streamlit as st
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

# Fix OpenMP conflict
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"

# Initialize CLIP model
device = "cuda" if torch.cuda.is_available() else "cpu"
model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32", use_fast=False)

# Directory where your images are stored
IMAGE_DIR = "C:/Users/user/Music/AI Images"  # <--- CHANGE THIS
EMBEDDINGS_FILE = "image_embeddings.npy"
PATHS_FILE = "image_paths.npy"

# Simple keyword expansions for suggestions
SUGGESTIONS = {
    "dog": ["puppy", "golden retriever", "labrador", "bulldog"],
    "flower": ["rose", "tulip", "hibiscus", "sunflower"],
    "car": ["sports car", "SUV", "convertible", "classic car"],
    "tree": ["oak", "pine", "cherry blossom", "willow"],
    "beach": ["tropical beach", "sunset beach", "rocky shore", "lagoon"],
    "mountain": ["snowy mountain", "rocky peak", "volcano", "hiking trail"],
    "bird": ["sparrow", "eagle", "peacock", "parrot"]
}

# Agent class to manage search behavior
class ImageSearchAgent:
    def __init__(self):
        self.index = None
        self.image_paths = []
        self.embeddings = None
        self.load_or_build_index()

    def load_or_build_index(self):
        if os.path.exists(EMBEDDINGS_FILE) and os.path.exists(PATHS_FILE):
            self.embeddings = np.load(EMBEDDINGS_FILE)
            self.image_paths = np.load(PATHS_FILE, allow_pickle=True)
            self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
            self.index.add(self.embeddings)
        else:
            self.index_images()

    def index_images(self):
        image_embeddings = []
        image_paths = []

        for root, dirs, files in os.walk(IMAGE_DIR):
            for file in files:
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    path = os.path.join(root, file)
                    try:
                        image = Image.open(path).convert("RGB")
                        inputs = processor(images=image, return_tensors="pt").to(device)
                        with torch.no_grad():
                            embedding = model.get_image_features(**inputs)
                        embedding = embedding.cpu().numpy().flatten()
                        image_embeddings.append(embedding)
                        image_paths.append(path)
                    except Exception as e:
                        print(f"Failed to process {path}: {e}")

        self.embeddings = np.vstack(image_embeddings)
        self.image_paths = np.array(image_paths)
        np.save(EMBEDDINGS_FILE, self.embeddings)
        np.save(PATHS_FILE, self.image_paths)

        self.index = faiss.IndexFlatL2(self.embeddings.shape[1])
        self.index.add(self.embeddings)

    def refresh_index_if_needed(self):
        current_files = []
        for root, dirs, files in os.walk(IMAGE_DIR):
            for file in files:
                if file.lower().endswith((".png", ".jpg", ".jpeg")):
                    current_files.append(os.path.join(root, file))
        if set(current_files) != set(self.image_paths):
            print("New images detected, rebuilding index...")
            self.index_images()

    def search_by_text(self, query, top_k=5):
        self.refresh_index_if_needed()
        inputs = processor(text=[query], return_tensors="pt", padding=True).to(device)
        with torch.no_grad():
            query_embedding = model.get_text_features(**inputs).cpu().numpy()
        D, I = self.index.search(query_embedding, top_k)
        return [self.image_paths[idx] for idx in I[0]]

    def suggest_query_modifications(self, query):
        suggestions = []
        for keyword, expansions in SUGGESTIONS.items():
            if keyword in query.lower():
                suggestions.extend(expansions)
        return suggestions

    def search_by_image(self, uploaded_image, top_k=5):
        self.refresh_index_if_needed()
        image = Image.open(uploaded_image).convert("RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)
        with torch.no_grad():
            query_embedding = model.get_image_features(**inputs).cpu().numpy()
        D, I = self.index.search(query_embedding, top_k)
        return [self.image_paths[idx] for idx in I[0]]

# Streamlit web UI
st.set_page_config(page_title="Local Image Search", layout="wide")
st.title("🖼️ Local Image Search Engine (Agentic AI)")

agent = ImageSearchAgent()

search_mode = st.radio("Choose search mode:", ("Text Search", "Image Search"))

if search_mode == "Text Search":
    query = st.text_input("Enter your search query:")
    if st.button("Search"):
        if query:
            suggestions = agent.suggest_query_modifications(query)
            if suggestions:
                st.write("🔎 **Did you mean:**")
                for suggestion in suggestions:
                    if st.button(f"Try '{suggestion}'"):
                        query = suggestion
                        break
            results = agent.search_by_text(query)
            st.subheader("Top Matches:")
            for path in results:
                st.image(path, caption=os.path.basename(path), use_container_width=True)
elif search_mode == "Image Search":
    uploaded_file = st.file_uploader("Upload an image to search for similar images", type=["png", "jpg", "jpeg"])
    if uploaded_file is not None:
        st.image(uploaded_file, caption="Uploaded Image", use_container_width=False)
        if st.button("Search"):
            results = agent.search_by_image(uploaded_file)
            st.subheader("Top Matches:")
            for path in results:
                st.image(path, caption=os.path.basename(path), use_container_width=True)

st.sidebar.info("This search runs completely locally without cloud dependencies.")
