import numpy as np

from src.Segment import Segment
from src.Video import Video
from .VideoReader import VideoReader
from .histograms import compute_histograms
import os

DATA_PATH = os.path.normpath(os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'data'))
SEGMENTS_PATH = os.path.join(DATA_PATH, "segments")
MOVIE_PATH = os.path.join(DATA_PATH, "movies")


def load_training_set(video_set: set):
    """
    Load and process all videos in provided training set.
    
    video_set: List of integers corresponding to video files (5 char left zero padded).
    """

    for i in video_set:
        # Int to movie name
        name = "{:05d}".format(i)

        # Process
        yield process_video(name)


def process_video(name: str) -> Video:
    display(name)
    video_path = os.path.join(MOVIE_PATH, name + ".mp4")
    segments_path = os.path.join(SEGMENTS_PATH, name + ".tsv")

    # Load movie in memory
    source_video = VideoReader()
    source_video.open(video_path)

    # Load segments file
    segment_data = np.genfromtxt(segments_path, delimiter="\t", skip_header=1, filling_values=1)

    # 4. Create video object and convert segments
    video = Video()
    frame_iter = source_video.get_frames()
    video.segments = np.apply_along_axis(lambda row: create_segment(frame_iter, row), arr=segment_data, axis=1)
    
    return video


def create_segment(video_frames, row: np.ndarray) -> Segment:
    """"
    Row layout: [startframe, starttime, endframe, endtime]
    """

    # Create new segment
    s = Segment(row[1], row[3], row[0], row[2])

    # Accumulate frames in segment
    framebuffer = []
    for _ in range(s.num_frames()): framebuffer.append(next(video_frames))

    # Generate histograms
    s.histograms = __generate_histograms(np.asarray(framebuffer))

    return s


def __generate_histograms(framebuffer: np.ndarray) -> np.ndarray:
#     print(framebuffer.shape)
#     print(framebuffer)
    
#     for first frame in buffer compute histograms
    return compute_histograms(framebuffer[0])
    
