import streamlit as st
from transformers import AutoModelForCausalLM, AutoTokenizer ,BitsAndBytesConfig
from langchain.prompts import PromptTemplate
from huggingface_hub import login
import json,torch
# Load model directly
#token ="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"



# Function to get response from Llama 2 model
def get_llama_response(input_text, no_words, blog_style):
    config_data = json.load(open(r"c:\Users\Sai\Desktop\My\dream\llm\blog_generator\config.json"))
    huggingface_token = config_data["hf_token"]
    #Quantization
    quant_config = BitsAndBytesConfig(
                                      load_in_4bit=True,
                                      bnb_4bit_compute_dtype=torch.bfloat16)

    tokenizer = AutoTokenizer.from_pretrained("meta-llama/Llama-2-7b-chat-hf",token=huggingface_token)
    tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-2-7b-chat-hf",device_map="auto",quantization_config=quant_config,
                                                token=huggingface_token,low_cpu_mem_usage=True)
    # Prompt Template
    template = """
        Write a blog for {blog_style} job profile for a topic {input_text}
        within {no_words} words.
    """
    
    prompt = PromptTemplate(input_variables=["blog_style", "input_text", 'no_words'],
                            template=template)
    
    # Generate the response from the Llama 2 model
    input_prompt = prompt.format(blog_style=blog_style, input_text=input_text, no_words=no_words)
    
    # Tokenize the input and generate output
    inputs = tokenizer(input_prompt, return_tensors="pt")
    outputs = model.generate(**inputs, max_new_tokens=256, temperature=0.01)

    # Decode the generated text
    response = tokenizer.decode(outputs[0], skip_special_tokens=True)
    
    return response

# Streamlit app configuration
st.set_page_config(page_title="Generate Blogs",
                   page_icon='🤖',
                   layout='centered',
                   initial_sidebar_state='collapsed')

st.header("Generate Blogs 🤖")

input_text = st.text_input("Enter the Blog Topic")

# Creating two more columns for additional fields
col1, col2 = st.columns([5, 5])

with col1:
    no_words = st.text_input('No of Words')
with col2:
    blog_style = st.selectbox('Writing the blog for',
                               ('Researchers', 'Data Scientist', 'Common People'), index=0)
    
submit = st.button("Generate")

# Final response
if submit:
    response = get_llama_response(input_text, no_words, blog_style)
    st.write(response)




