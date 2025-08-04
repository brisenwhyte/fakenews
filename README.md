There are 3 setups needed before running the application

1. DeepFake Backend
2. Article Detection Bacxkend 
3. Front end 

1. DeepFake Backend 
 navigate to Deepfake\Django Application
and run 
1. python -m venv venv
2. venv\Scripts\activate
3.pip install -r require.txt
4. pip install dlib
 
if you are using a conda environment there might be a conflict in versions
you can run 

1. conda deactivate
2.conda create --name deepfake_env python=3.9
3.conda activate deepfake_env
4. pip install -r require.txt
5. conda install -c conda-forge dlib



2. Article Detection Backend 
set the gemini api key 
Navigate to 
1. fake article 2
in command prompt run : SET GOOGLE_API_KEY="your api key"
go to google AI Studio and generate an api key  
2. run: python app.py


3.Front end 

Navigate to:
UI\fake-news-detection-system

run: npm start 
