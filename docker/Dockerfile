FROM python:3.10-slim

WORKDIR /app

# Copy requirements and install
COPY cicd/requirements.txt /app/requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# Copy app
COPY app /app/app

EXPOSE 5000
CMD gunicorn -b 0.0.0.0:$PORT app.app:app
