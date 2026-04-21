echo "Going to parent directory..."
cd ..

echo "Creating virtual environment..."
python3 -m venv myenv

echo "Activating virtual environment..."
source myenv/bin/activate

echo "Upgrading pip..."
pip install --upgrade pip

echo "Installing dependencies..."
pip install -r requirements.txt

echo "Setup complete"

