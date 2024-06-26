# Use Python 3.9 base image
FROM python:3.9

# Set the working directory in the container
WORKDIR /app2

# Copy the rest of your application code
COPY . /app2

# Install Python dependencies
RUN pip install -r requirements.txt

# Expose the port your app runs on (if needed)
EXPOSE 8501

# Command to run the application
CMD ["streamlit", "run", "app2.py"]
