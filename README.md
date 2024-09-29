# Qwen2-Colpali-OCR


This Streamlit application demonstrates a Multimodal Retrieval-Augmented Generation (RAG) system using the Qwen2-VL model and a custom RAG implementation. It allows users to upload images and ask questions about them, combining visual and textual information to generate responses.

## Features

- Image upload or selection of an example image
- Text-based querying of uploaded images
- Multimodal RAG processing using custom RAG model and Qwen2-VL
- Adjustable response length

## Prerequisites

- Python 3.8+
- CUDA-compatible GPU (recommended for optimal performance)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/your-username/multimodal-rag-app.git
   cd multimodal-rag-app
   ```

2. Create a virtual environment:
   ```
   python -m venv venv
   source venv/bin/activate  # On Windows, use `venv\Scripts\activate`
   ```

3. Install the required packages:
   ```
   pip install -r requirements.txt
   ```

## Running the Application Locally

1. Ensure you're in the project directory and your virtual environment is activated.

2. Run the Streamlit app:
   ```
   streamlit run app.py
   ```

3. Open a web browser and navigate to the URL provided by Streamlit (usually `http://localhost:8501`).

## Usage

1. Choose to upload an image or use the example image.
2. If uploading, select an image file (PNG, JPG, or JPEG).
3. Enter a text query about the image in the provided input field.
4. Adjust the maximum number of tokens for the response using the slider.
5. View the generated response based on the image and your query.

## Deployment

This application can be deployed on various platforms that support Streamlit apps. Here are general steps for deployment:

1. Ensure all dependencies are listed in `requirements.txt`.
2. Choose a deployment platform (e.g., Streamlit Cloud, Heroku, or a cloud provider like AWS or GCP).
3. Follow the platform-specific deployment instructions, which typically involve:
   - Connecting your GitHub repository to the deployment platform
   - Configuring environment variables if necessary
   - Setting up any required build processes

Note: For optimal performance, deploy on a platform that provides GPU support.

## Disclaimer

The app is configured to run on CPU by default, which may result in slower processing times. For better performance, it's recommended to run the app locally on a machine with GPU support.

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

[Specify your license here]

## Acknowledgments

- This project uses the [Qwen2-VL model](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) from Hugging Face.
- The custom RAG implementation is based on the [colpali model](https://huggingface.co/vidore/colpali).
