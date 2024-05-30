# Select base image for the project
FROM python:3.12

# Choose what directory should be a working directory
# If the directory does not exist then Docker will create it
WORKDIR /app

# Copy project files into working directory
COPY /app /app

# Copy our local file with requirements into Docker Image
COPY requirements.txt ./

# Install all the requirements
RUN pip install --no-cache-dir -r requirements.txt

# Open port for Docker
EXPOSE 3001

# Run flask
CMD streamlit run app.py --server.port=3001