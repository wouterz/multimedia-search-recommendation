import pandas as pd
import numpy as np
from .VideoReader import VideoReader
import os

DATA_PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'))
SEGMENTS_PATH = os.path.join(DATA_PATH, "segments")
MOVIE_PATH = os.path.join(DATA_PATH, "movies")


def load_training_set(video_set: set):
    
    for i in video_set:
    
        # Int to movie name
        name = "{:05d}".format(i)
        
        # Process
        process_video(name)
    

def process_video(name: str):
    
    # TODO: convert to generator
    
    video_path = os.path.join(MOVIE_PATH, name+".mp4")
    segments_path = os.path.join(SEGMENTS_PATH, name+".tsv")
    
    # 1. Load movie in memory
    video = VideoReader()
    video.open(video_path)
    
    # 2. Load segments file
    
    
    
    # 3. 
    