import streamlit as st
from groq import Groq
import base64
from PIL import Image as PILImage
import os
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Get Groq API key
groq_api_key = os.getenv("GROQ_API_KEY")

# Initialize Groq API client
client = Groq(api_key=groq_api_key)

# Corrected model names
vision_model = "llama-3.2-11b-vision-preview"  # Replacing LLaVA
llama31_model = "llama-3.1-70b-versatile"  # For text generation

# Function to resize image if too large
def resize_image(image, max_size=(800, 800)):
    if image.size[0] > max_size[0] or image.size[1] > max_size[1]:
        image.thumbnail(max_size)
    return image

# Function to encode image to base64
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

# Function to generate image description using Llama 3.2 Vision
def image_to_text(client, model, base64_image, prompt):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"},
                    },
                ],
            }
        ],
        model=model,
    )

    return chat_completion.choices[0].message.content

# Function to generate short story using Llama 3.1
def short_story_generation(client, image_description):
    chat_completion = client.chat.completions.create(
        messages=[
            {
                "role": "system",
                "content": "You are a children's book author. Write a short story about the scene depicted in this image.",
            },
            {"role": "user", "content": image_description},
        ],
        model=llama31_model,
    )

    return chat_completion.choices[0].message.content

# Streamlit app title
st.title("Llama 3.2 Vision & Llama 3.1: Image Description and Story Generator")

# Image upload section
uploaded_image = st.file_uploader("Upload an image", type=["png", "jpg", "jpeg"])

if uploaded_image:
    image = PILImage.open(uploaded_image)
    resized_image = resize_image(image)

    # Save resized image
    if not os.path.exists("temp_images"):
        os.makedirs("temp_images")

    image_path = os.path.join("temp_images", uploaded_image.name)
    resized_image.save(image_path)

    # Display the image
    st.image(resized_image, caption="Uploaded Image (Resized)", use_column_width=True)

    # Encode image
    base64_image = encode_image(image_path)

    # Generate image description
    st.write("Generating image description using Llama 3.2 Vision...")
    description_prompt = "Describe this image in detail, including key elements and actions."
    image_description = image_to_text(client, vision_model, base64_image, description_prompt)

    st.write("### Image Description")
    st.write(image_description)

    # Generate short story
    st.write("Generating short story using Llama 3.1...")
    short_story = short_story_generation(client, image_description)

    st.write("### Generated Story")
    st.write(short_story)
