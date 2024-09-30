import torch
import torch.nn.functional as F
import numpy as np
import streamlit as st
from PIL import Image
from streamlit_drawable_canvas import st_canvas
from torchvision.transforms import InterpolationMode
from torchvision.transforms.v2 import Resize, Normalize, Compose, ToImage, ToDtype, RGB
from model import TripletModel

model = TripletModel("convnext")
model.load_state_dict(torch.load('model.pth', map_location=torch.device('cpu'), weights_only=True))
model.eval()

transforms = Compose([
    RGB(),
    Resize((224, 224), interpolation=InterpolationMode.BILINEAR),
    ToImage(),
    ToDtype(torch.float32, scale=True),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

gallery_embeddings = np.load('imagenet_a_emb.npy')  # Shape (N, embedding_dim)
# gallery_embeddings = torch.from_numpy(gallery_embeddings)
gallery_image_paths = np.load('imagenet_a_paths.npy')  # Shape (N,)

def find_similar_embeddings(query_embedding, gallery_embeddings, top_k=10):
    query_embedding = torch.from_numpy(query_embedding).float()
    gallery_embeddings = torch.from_numpy(gallery_embeddings).float()
    # Compute cosine similarity between the query and all gallery embeddings
    similarities = F.pairwise_distance(query_embedding, gallery_embeddings)
    # Get the indices of the top_k most similar embeddings
    top_k_indices = torch.sort(similarities)[1][:top_k]
    return top_k_indices

st.title("TEST APP")

# Create a canvas component
canvas_result = st_canvas(
    fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
    stroke_width=3,
    stroke_color="#000000",
    background_color="#EEEEEE",
    height=512,
    width=512,
    drawing_mode="freedraw",
    key="canvas",
)   

if st.button("Generate Embedding"):
    if canvas_result.image_data is not None:
        st.write("generating embeddings")
        # Convert the drawn image to a PIL image
        img = Image.fromarray((canvas_result.image_data[:, :, :3]).astype(np.uint8))

        # Preprocess the image
        img_tensor = transforms(img).unsqueeze(0)  # Add batch dimension

        # Pass the image to the model to get the embedding
        with torch.no_grad():
            query_embedding = model.get_embedding(img_tensor).numpy()

        top_k_indices = find_similar_embeddings(query_embedding, gallery_embeddings, top_k=10)

        # Display the 10 most similar images
        st.write("Top 10 Similar Images:")
        cols = st.columns(5)  # Create columns for displaying images
        cols2 = st.columns(5)

        for i, idx in enumerate(top_k_indices):
            # Load and display the corresponding image from the gallery
            # similar_img = Image.open(os.path.join("..", gallery_image_paths[idx]))
            similar_img = gallery_image_paths[idx]
            if i < 5:
                cols[i % 5].image(similar_img, width=100, caption=str(i + 1))  # Display the image in one of the 5 columns
            else:
                cols2[i % 5].image(similar_img, width=100, caption=str(i + 1))
    else:
        st.write("Please draw something first!")
