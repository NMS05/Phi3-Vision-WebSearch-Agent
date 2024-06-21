from PIL import Image
import base64
from io import BytesIO

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import torch
from transformers import AutoModelForCausalLM, AutoProcessor


# Define the FastAPI app
app = FastAPI()

# converts string to PIL.Image object
def base64_string_to_PIL_Image(base64_str):
    byte_data = base64.b64decode(base64_str)
    return Image.open(BytesIO(byte_data))

"""
Pydantic is the most widely used data validation library for Python.
FastAPI cannot work without Pydantic
"""
# Define a request model
class ImageTextRequest(BaseModel):
    str_image: str
    user_query: str

# Define a response model
class TextResponse(BaseModel):
    vlm_response: str

"""
Phi3 Vision Model
"""
device = torch.device("cuda:2")
model_id = "microsoft/Phi-3-vision-128k-instruct"
model = AutoModelForCausalLM.from_pretrained(model_id, trust_remote_code=True, torch_dtype="auto", _attn_implementation='flash_attention_2').to(device)
model.eval()
processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True)


"""
FastAPI server for Phi3 Vision
"""
# Endpoint to process text and return the model's prediction
@app.post('/predict', response_model=TextResponse)
def predict(request: ImageTextRequest):
    try:

        if request.str_image == 'No Image Provided':
            input_image = None
        else:
            input_image = base64_string_to_PIL_Image(request.str_image)
        user_prompt = request.user_query

        # convert the user prompt to phi3v prompt format and tokenize them
        messages = [{"role": "user", "content": f"<|image_1|>\n{user_prompt}"}]
        prompt = processor.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)

        if input_image == None:
            inputs = processor(prompt, images=None, return_tensors="pt").to(device)
        else:
            inputs = processor(prompt, [input_image], return_tensors="pt").to(device)

        # generate response
        with torch.no_grad():
            generate_ids = model.generate(**inputs, max_new_tokens=200, eos_token_id=processor.tokenizer.eos_token_id)
            generate_ids = generate_ids[:, inputs['input_ids'].shape[1]:] # remove input tokens
            response = processor.batch_decode(generate_ids, skip_special_tokens=True, clean_up_tokenization_spaces=False)[0]
        return TextResponse(vlm_response=response)
    
    # this makes sure that the error message (if any) is sent to the client via API
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

"""
To run the server, `uvicorn Phi3_Vision_Server:app --host 127.0.0.1 --port 8001`
"""