# Qwen2-Colpali-OCR


This application demonstrates a Multimodal Retrieval-Augmented Generation (RAG) system using the Qwen2-VL model and a custom RAG implementation. It allows users to upload images and ask questions about them, combining visual and textual information to generate responses.


It is deployed here on HuggingFace Spaces https://huggingface.co/spaces/clayton07/qwen2-colpali-ocr

## Prerequisites

- Python 3.8+
- Pytorch 2.4.1
- Torchvision 0.19.1
- Qwen V1
- Byaldi
- CUDA-compatible GPU (recommended for optimal performance)

## Installation

1. Clone the repository:
   ```
   git clone https://github.com/Claytonn7/qwen2-colpali-ocr.git
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
3. Enter a single keyword in the provided input field.
4. Adjust the maximum number of tokens for the response using the slider.
5. View the extracted text from the image, with the searched keyword highlighted. Example screenshot [here](https://github.com/Claytonn7/qwen2-colpali-ocr/blob/main/examples-app/6-keyword-highlight2.jpg)

NB: Check the examples-app directory on this repo for more example screenshots.

## Disclaimer

The app utilizes the free tier of HuggingFace Spaces, which only has support for CPU, resulting in very slow processing times. For optimal performance, it's recommended to run the app locally on a machine with GPU support.

## Acknowledgments

- This project uses the [Qwen2-VL model](https://huggingface.co/Qwen/Qwen2-VL-2B-Instruct) from Hugging Face.
- The [byaldi](https://github.com/AnswerDotAI/byaldi) implementation of the [colpali model](https://huggingface.co/vidore/colpali).

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the GPL-2.0 License
