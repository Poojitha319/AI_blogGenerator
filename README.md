# AI_blogGenerator
# Blog Generator with Llama 2

## Overview

This project is a Streamlit application that generates blog posts based on user-defined topics and styles using the Llama 2 model from Hugging Face. The application allows users to specify the number of words for the blog post and select a specific audience, making it versatile for different writing needs.

## Features

- Generate blog content tailored to various job profiles:
  - Researchers
  - Data Scientists
  - Common People
- Utilize the Llama 2 model for high-quality text generation.
- Customize the length of the blog post.
- User-friendly interface built with Streamlit.

## Requirements

To run this application, ensure you have the following installed:

- Python 3.7 or higher
- Streamlit
- Transformers
- Hugging Face Hub
- PyTorch
- BitsAndBytes

## Setup Instructions

1. **Clone the repository:**
   ```bash
   git clone https://github.com/Poojitha319/AI_blogGenerator
   cd blog_generator
2.**Install required packages:**
```bash
   pip install streamlit transformers huggingface_hub torch bitsandbytes
```
3.**Create configuration file:**
Create a file named config.json inside the blog_generator directory with the following content:
```bash
   {
  "hf_token": "YOUR_HUGGINGFACE_TOKEN"
   }
```
4.**Run the streamlit app:**
```bash
   streamlit run app.py
```
## Acknowledgements
-[Hugging face ](https://huggingface.co/) for providing the Llama 2 model.
-[Streamlit](https://streamlit.io/) for making it easy to create web applications for machine learning projects.


