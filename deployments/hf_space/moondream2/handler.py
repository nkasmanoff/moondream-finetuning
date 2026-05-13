from transformers import AutoModelForCausalLM, AutoTokenizer
from PIL import Image
import torch
from io import BytesIO
import base64

class EndpointHandler:
    def __init__(self, model_dir):
        self.model_id = "vikhyatk/moondream2"
        self.model = AutoModelForCausalLM.from_pretrained(self.model_id, trust_remote_code=True)
        self.tokenizer = AutoTokenizer.from_pretrained("vikhyatk/moondream2", trust_remote_code=True)

        # Check if CUDA (GPU support) is available and then set the device to GPU or CPU
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def preprocess_image(self, encoded_image):
        """Decode and preprocess the input image."""
        decoded_image = base64.b64decode(encoded_image)
        img = Image.open(BytesIO(decoded_image)).convert("RGB")
        return img

    def __call__(self, data):
        """Handle the incoming request."""
        try:
            # Extract the inputs from the data
            inputs = data.pop("inputs", data)
            input_image = inputs['image']
            question = inputs.get('question', "move to the red ball")

            # Preprocess the image
            img = self.preprocess_image(input_image)

            # Perform inference
            enc_image = self.model.encode_image(img).to(self.device)
            answer = self.model.answer_question(enc_image, question, self.tokenizer)

            # If the output is a tensor, move it back to CPU and convert to list
            if isinstance(answer, torch.Tensor):
                answer = answer.cpu().numpy().tolist()

            # Create the response
            response = {
                "statusCode": 200,
                "body": {
                    "answer": answer
                }
            }
            return response
        except Exception as e:
            # Handle any errors
            response = {
                "statusCode": 500,
                "body": {
                    "error": str(e)
                }
            }
            return response