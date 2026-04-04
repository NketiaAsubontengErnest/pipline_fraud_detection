import sys
import os

# Add the parent directory to sys.path so we can import from src
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
# Add the src directory as well so internal imports within src like 'from features import ...' work
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from src.reporting_api import app
