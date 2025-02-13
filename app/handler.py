import runpod
import os
from dotenv import load_dotenv
import torch

load_dotenv()

runpod.api_key = os.getenv("RUNPOD_API_KEY")

endpoint = runpod.Endpoint("SERVER_ENDPOINT_ID")

# Fetching all available endpoints
endpoints = runpod.get_endpoints()

# Displaying the list of endpoints
print(endpoints)

input_payload = {"input": {"prompt": "Hello, World! The runpod servless endpoint is working."}}

try:
    endpoint = runpod.Endpoint("SERVER_ENDPOINT_ID")
    run_request = endpoint.run(input_payload)

    # Initial check without blocking, useful for quick tasks
    status = run_request.status()
    print(f"Initial job status: {status}")

    if status != "COMPLETED":
        # Polling with timeout for long-running tasks
        output = run_request.output(timeout=60)
    else:
        output = run_request.output()
    print(f"Job output: {output}")
except Exception as e:
    print(f"An error occurred: {e}")


# HUGGING_FACE_LOGIN = os.getenv("HUGGING_FACE_LOGIN")

# from diffusers import FluxPipeline
# from huggingface_hub import login

# # Authenticate with Hugging Face
# login(HUGGING_FACE_LOGIN)

# pipe = FluxPipeline.from_pretrained("black-forest-labs/FLUX.1-dev", torch_dtype=torch.bfloat16)
# #pipe.enable_model_cpu_offload() #save some VRAM by offloading the model to CPU. Remove this if you have enough GPU power

# prompt = "A puppy holding a sign that says hello world"
# image = pipe(
#     prompt,
#     height=1024,
#     width=1024,
#     guidance_scale=3.5,
#     num_inference_steps=50,
#     max_sequence_length=512,
#     generator=torch.Generator("cpu").manual_seed(0)
# ).images[0]
# image.save("flux-dev.png")
