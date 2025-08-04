This document outlines the necessary steps to set up and run the application. The application consists of three main components: a DeepFake Backend, an Article Detection Backend, and a Front end.

1. DeepFake Backend
This backend is responsible for DeepFake detection.

Standard Python Environment
Navigate to the Directory:
cd Deepfake\Django Application

Create a Virtual Environment:
python -m venv venv

Activate the Virtual Environment:
venv\Scripts\activate

Install Dependencies:
pip install -r require.txt

Install Dlib:
pip install dlib

Conda Environment (Alternative)
If you encounter version conflicts, a Conda environment is recommended.

Deactivate Existing Conda Environment:
conda deactivate

Create a New Conda Environment:
conda create --name deepfake_env python=3.9

Activate the New Environment:
conda activate deepfake_env

Install Dependencies:
pip install -r require.txt

Install Dlib (using Conda):
conda install -c conda-forge dlib

2. Article Detection Backend
This backend handles the detection of fake articles.

Setup and Running
Set the Gemini API Key:

Navigate to the fake article 2 directory.

Obtain an API key from Google AI Studio.

Run the following command in the command prompt, replacing "your api key" with your actual key:
SET GOOGLE_API_KEY="your api key"

Run the Application:
python app.py

3. Front End
This is the user interface for the application.

Setup and Running
Navigate to the Directory:
cd UI\fake-news-detection-system

Start the Application:
npm start
