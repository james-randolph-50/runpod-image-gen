import runpod
import os
from dotenv import load_dotenv
import torch

load_dotenv()

runpod.api_key = os.getenv("RUNPOD_API_KEY")


HUGGING_FACE_LOGIN = os.getenv("HUGGING_FACE_LOGIN")

print(torch.version.cuda)  # Should show CUDA version
print(torch.cuda.device_count())  # Should be > 0.

from diffusers import FluxPipeline
from huggingface_hub import login

# Authenticate with Hugging Face
login(HUGGING_FACE_LOGIN)

print(torch.cuda.is_available())

pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
#pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

prompt = "A puppy holding a sign that says hello world"
image = pipe(
    prompt,
    height=1024,
    width=1024,
    guidance_scale=3.5,
    num_inference_steps=50,
    max_sequence_length=512,
    generator=torch.Generator("cpu").manual_seed(0)
).images[0]
image.save("flux-dev.png")
