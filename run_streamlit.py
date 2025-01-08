import os
import sys
from pathlib import Path

# Add the project root to Python path
project_root = Path(__file__).parent.absolute()
sys.path.insert(0, str(project_root))

# Format the path to the Streamlit app
streamlit_app_path = str(project_root / "app" / "frontend" / "streamlit_app.py")

# Run Streamlit with the correct module format
if __name__ == "__main__":
    # Set PYTHONPATH environment variable
    os.environ["PYTHONPATH"] = str(project_root)
    # Run Streamlit
    os.system(f"PYTHONPATH={str(project_root)} poetry run streamlit run {streamlit_app_path}") 