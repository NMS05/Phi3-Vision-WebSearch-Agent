import requests
import base64
from io import BytesIO

class My_VLM_APIs():

    def __init__(self, model_name) -> None:

        currently_available_apis = ['mini_cpm_llama3v','phi3_vision']

        # define the URL for the FastAPI servers
        # localhost defaults to 127.0.0.1
        if model_name == 'mini_cpm_llama3v':
            self.server_url = 'http://localhost:8000/predict'
        elif model_name == 'phi3_vision':
            self.server_url = 'http://localhost:8001/predict'
        else:
            print("\n API unavailable")
            print(f"\n Currently available APIs are {currently_available_apis}")
            exit()

    # you cannot send the image directly to the FastAPI server as a json - <TypeError: Object of type Image is not JSON serializable>
    # this snippet converts a PIL image to a string that can be sent via a json
    def  PIL_Image_to_base64_string(self, PIL_image):
        with PIL_image as img:
            # Create a BytesIO buffer and save the image data to it
            with BytesIO() as buffer:
                img.save(buffer, 'JPEG')
                # Get the byte data from the buffer
                byte_data = buffer.getvalue()
        # Convert the byte data to a base64 string
        base64_str = base64.b64encode(byte_data).decode()
        return base64_str
    
    # Function to send a request to FastAPI server and get response
    def get_response(self, image, query):

        if image == None:
            image_as_string = 'No Image Provided'
        else:
            image_as_string = self.PIL_Image_to_base64_string(image)

        # send request to server
        response = requests.post(self.server_url, json={'str_image': image_as_string, 'user_query': query})
        # Raise an error if the request was unsuccessful
        response.raise_for_status()

        # return the response
        output = response.json()
        return output['vlm_response']