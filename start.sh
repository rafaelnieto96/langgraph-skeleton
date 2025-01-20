# Activate virtual environment
./venv/Scripts/activate

# Install dependencies
pip install -r requirements.txt

# Run the application
python ./src/main.py

# Run tests
pytest  .\tests\test_graph.py