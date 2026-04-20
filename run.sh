echo "Creating virtual environment..."
python3 -m venv myenv

echo "Activating virtual environment..."
source myenv/bin/activate

echo "Installing dependencies..."
pip install -r requirements.txt

