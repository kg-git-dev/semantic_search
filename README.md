# Create and activate the virtual environment
python3 -m venv venv
. venv/bin/activate

# Install dependencies from requirements.txt
pip install -r requirements.txt

# Create a pickle file
python3 generateEncodings.py

# run the container
gunicorn --reload -b  0.0.0.0:8000 app:app