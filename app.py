import streamlit as st
import os
import torch
from byaldi import RAGMultiModalModel
from transformers import Qwen2VLForConditionalGeneration, AutoTokenizer, AutoProcessor
from qwen_vl_utils import process_vision_info
import re

# Check for CUDA availability
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Caching the model loading
@st.cache_resource
def load_rag_model():
    return RAGMultiModalModel.from_pretrained("vidore/colpali")

@st.cache_resource
def load_qwen_model():
    return Qwen2VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen2-VL-2B-Instruct",
        trust_remote_code=True,
        torch_dtype=torch.bfloat16
    ).to(device).eval()

@st.cache_resource
def load_processor():
    return AutoProcessor.from_pretrained("Qwen/Qwen2-VL-2B-Instruct", trust_remote_code=True)

# Load models
RAG = load_rag_model()
model = load_qwen_model()
processor = load_processor()

st.title("Multimodal RAG App")

st.warning("⚠️ Disclaimer: This app is currently running on CPU, which may result in slow processing times (even loading the image may take more than 10 minutes). For optimal performance, download and run the app locally on a machine with GPU support.")

# Add download link
st.markdown("[📥 Download the app code](https://github.com/Claytonn7/qwen2-colpali-ocr)")

# Initialize session state for tracking if index is created
if 'index_created' not in st.session_state:
    st.session_state.index_created = False

# File uploader
image_source = st.radio("Choose image source:", ("Upload an image", "Use example image"))

if image_source == "Upload an image":
    uploaded_file = st.file_uploader("Choose an image file", type=["png", "jpg", "jpeg"])
else:
    # Use a pre-defined example image
    example_image_path = "hindi-qp.jpg"
    uploaded_file = example_image_path

if uploaded_file is not None:
    # If using the example image, no need to save it
    if image_source == "Upload an image":
        with open("temp_image.png", "wb") as f:
            f.write(uploaded_file.getvalue())
        image_path = "temp_image.png"
    else:
        image_path = uploaded_file

    if not st.session_state.index_created:
        # Initialize the index for the first image
        RAG.index(
            input_path=image_path,
            index_name="temp_index",
            store_collection_with_index=False,
            overwrite=True
        )
        st.session_state.index_created = True
    else:
        # Add to the existing index for subsequent images
        RAG.add_to_index(
            input_item=image_path,
            store_collection_with_index=False
        )

    st.image(uploaded_file, caption="Uploaded Image", use_column_width=True)

    # Text query input
    text_query = st.text_input("Enter a single word to search for:")
    extract_query = "extract text from the image"

    max_new_tokens = st.slider("Max new tokens for response", min_value=100, max_value=1000, value=100, step=10)

    if text_query:
        with st.spinner(
                f'Processing your query... This may take a while due to CPU processing. Generating up to {max_new_tokens} tokens.'):
            # Perform RAG search
            results = RAG.search(text_query, k=2)

            # Process with Qwen2VL model
            messages = [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image_path,
                        },
                        {"type": "text", "text": extract_query},
                    ],
                }
            ]
            text = processor.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            image_inputs, video_inputs = process_vision_info(messages)
            inputs = processor(
                text=[text],
                images=image_inputs,
                videos=video_inputs,
                padding=True,
                return_tensors="pt",
            )
            inputs = inputs.to(device)
            generated_ids = model.generate(**inputs, max_new_tokens=max_new_tokens)  # Using the slider value here
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs.input_ids, generated_ids)
            ]
            output_text = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )

            def highlight_text(text, query):
                if not query.strip():
                    return text

                escaped_query = re.escape(query)
                pattern = r'\b' + escaped_query + r'\b'

                def replacer(match):
                    return f'<span style="background-color: green;">{match.group(0)}</span>'

                highlighted_text = re.sub(pattern, replacer, text, flags=re.IGNORECASE)
                return highlighted_text

        # Display results
        highlighted_output = highlight_text(output_text[0], text_query)

        # Display results
        st.subheader("Extracted Text (with query highlighted):")
        st.markdown(highlighted_output, unsafe_allow_html=True)
        # st.subheader("Results:")
        # st.write(output_text[0])


    # Clean up temporary file
    if image_source == "Upload an image":
        os.remove("temp_image.png")
else:
    st.write("Please upload an image to get started.")
