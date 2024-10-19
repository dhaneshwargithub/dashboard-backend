In order to execute the forecast in the dashboard, follow these steps:
After extracting the zip files, download Anaconda Navigator.
Execute the following commands:
conda create -n myenv python=3.8 (to create the environment)
conda activate myenv (to activate the environment)
pip install pandas uvicorn statsmodels fastapi (to install the required libraries)
To run the server, use the command: uvicorn main:app --reload
