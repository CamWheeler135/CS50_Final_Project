''' Code that plots the performance of the algorithm. '''
from pathlib import Path

def create_directory(directory_path: str):
    ''' Makes a folder in the directory to place all of the plots. '''
    
    Path(directory_path).mkdir(parents=True, exist_ok=True)

