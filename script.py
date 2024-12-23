from fastapi import FastAPI
from pydantic import BaseModel
import uvicorn
from transformers import GPT2Tokenizer, GPT2LMHeadModel, pipeline

# Initialize FastAPI app
app = FastAPI()


# Load the fine-tuned model and tokenizer for inference

model = GPT2LMHeadModel.from_pretrained('fine_tuned_gpt_model')
tokenizer = GPT2Tokenizer.from_pretrained('fine_tuned_gpt_model')

# Create a text generation pipeline
generator = pipeline("text-generation", model=model, tokenizer=tokenizer)

# Pydantic model for input data validation
class InputData(BaseModel):
    text: str

# FastAPI route for testing the server
@app.get("/")
def read_root():
    return {"message": "Model is running and ready to process requests!"}

# FastAPI route for text generation
@app.post("/generate/")
def generate_text(data: InputData):
    # Generate text using the fine-tuned model
    generated = generator(data.text, max_length=50, num_return_sequences=1)
    return {"generated_text": generated[0]["generated_text"]}

if __name__ == "__main__":
    # Run the FastAPI app with uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
