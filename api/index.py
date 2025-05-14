import sys
import os
from pathlib import Path
from app import app

# Add the parent directory to sys.path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

# This is necessary for Vercel serverless functions
app.debug = False

handler = app

# This line is required for Vercel
if __name__ == "__main__":
    app.run()