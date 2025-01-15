FROM python:3.11-slim

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Expose the port the app runs on
EXPOSE 8000

# Command to run the API server
CMD ["uvicorn", "chatbot:app", "--host", "0.0.0.0", "--port", "8000"]