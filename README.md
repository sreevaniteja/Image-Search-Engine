# Image-Search-Engine
A web application which can be used to search through personal images without uploading to drive or internet. Created as part of Microsoft AI SkillsFest 2025 Hackathon

# 🖼️ Local AI Image Search Engine (Agentic AI)

A powerful, private **local image search engine** that lets you find images from your personal collection by typing text queries or uploading example images.

This app uses **CLIP** models to generate image and text embeddings and **FAISS** for fast similarity search. It features an **Agentic AI** design — meaning it automatically refreshes the image index when new files are added and intelligently suggests better search prompts when the user input is vague.

Built with **Streamlit**, the entire system runs **offline** for maximum privacy and efficiency.

---

## ✨ Features

- 🔍 **Text-based Search**: Find images by entering a description.
- 📷 **Image-based Search**: Upload an image to find similar ones.
- 🧠 **Agentic Behavior**:
  - Auto-refreshes database if new images are added.
  - Suggests query improvements for better search results.
- ⚡ **Fast and Local**: No internet connection required.
- 🛡️ **Private**: All computation happens locally on your machine.

---

## 🚀 How to Run

1. Install dependencies:
   ```bash
   pip install torch torchvision faiss-cpu transformers streamlit Pillow
   ```

2. Set your local image directory:
   Open `script.py` and set:
   ```python
   IMAGE_DIR = "path_to_your_image_folder"
   ```

3. Run the app:
   ```bash
   streamlit script.py
   ```

4. Open your browser at `http://localhost:8501` and start searching!

---

## 🛠 Tech Stack

- **Python 3.10+**
- **Hugging Face Transformers** (CLIP Model)
- **FAISS** (Similarity Search)
- **Streamlit** (Web Interface)
- **PIL (Pillow)** (Image Handling)

---

## 📚 Future Improvements

- Batch image indexing for faster startup.
- Fuzzy query matching with local LLMs.
- Multi-modal search combining text + image inputs.
- Tag-based filtering (e.g., show only landscape or portrait images).

---

## ⚡ Quick Demo

> - Type `"dog"` → Get suggestions like `"puppy"`, `"golden retriever"`, `"labrador"`.
> - Upload a picture of a flower → Find similar flower images from your local library.

---

## 📩 Contributions

Feel free to suggest improvements, raise issues, or fork the project if you'd like to extend it! 🚀

---
