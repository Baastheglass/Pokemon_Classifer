from flask import Flask, render_template, request, jsonify
import os
import sys
from pathlib import Path

# Add the parent directory to sys.path
parent_dir = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(parent_dir))

from app import app

# This is necessary for Vercel serverless functions
app.debug = False

# This is the handler Vercel requires
handler = app

# This is only used for local development
if __name__ == "__main__":
    app.run()