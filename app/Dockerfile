# Use a PyTorch CUDA image with Python 3.9
FROM pytorch/pytorch:2.1.0-cuda12.1-cudnn8-runtime

WORKDIR /app

# Copy dependency list and install inside the final image
COPY requirements.txt .  
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY . .  

CMD ["python", "handler.py"]


# Copy dependencies from builder stage
# COPY --from=builder /usr/local/lib/python3.9/site-packages /usr/local/lib/python3.9/site-packages
# COPY --from=builder /usr/local/bin /usr/local/bin

