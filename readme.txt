Tested on Python 3.7 (Windows 10)

Install (from root directory):
   # Create Virtual Environment
   * python -m venv env
   * env/scripts/activate.bat
   # Install the system and all dependencies
   * pip install -e bsrs  # Installs 
   # Go to https://pytorch.org/ to install appropriate PyTorch version
   # GPUs supported if CUDA version installed

Running:
   * python code/biometrics.py
N.B.1. Code may take 20 minutes to run the first time if no GPU support.
N.B.2. Results should approximately match report, NNs have degree of non-determininsm.

Place in root level with folder name 'image_set'.
First execution will download all neural networks.