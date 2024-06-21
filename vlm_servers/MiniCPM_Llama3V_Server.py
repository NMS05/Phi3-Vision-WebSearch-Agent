from PIL import Image
import base64
from io import BytesIO

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

import torch
from transformers import AutoModel, AutoTokenizer


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
MiniCPM-Llama3-V 2.5
"""
# Load the model and tokenizer
model = AutoModel.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True, torch_dtype=torch.float16)
model = model.to(device='cuda')
tokenizer = AutoTokenizer.from_pretrained('openbmb/MiniCPM-Llama3-V-2_5', trust_remote_code=True)
model.eval()


"""
FastAPI server for MiniCPM-Llama3-V 2.5
"""
# Endpoint to process text and return the model's prediction
@app.post('/predict', response_model=TextResponse)
def predict(request: ImageTextRequest):
    try:

        assert request.str_image != 'No Image Provided', "MiniCPM-Llama3-V always expects an image as input"
        image = base64_string_to_PIL_Image(request.str_image)
        question = request.user_query
        message = [{'role': 'user', 'content': question}]

        
        # Get the model's predictions
        with torch.no_grad():
            mini_cpm_output = model.chat(
                    image=image,
                    msgs=message,
                    tokenizer=tokenizer,
                    sampling=True, # if sampling=False, beam_search will be used by default
                    temperature=0.1,
                    # system_prompt='' # pass system_prompt if needed
                )
        return TextResponse(vlm_response=mini_cpm_output)
    
    # this makes sure that the error message (if any) is sent to the client via API
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))

"""
To run the server, `uvicorn MiniCPM_Llama3V_Server:app --host 127.0.0.1 --port 8000`
"""