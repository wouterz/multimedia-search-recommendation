import numpy as np


class Segment:
    movie_id: str
    time_start: float
    time_end: float
    frame_start: int
    frame_end: int

    histograms: list

    def __init__(self, movie_id, time_start, time_end, frame_start, frame_end) -> None:
        self.movie_id = movie_id
        self.time_start = time_start
        self.time_end = time_end
        self.frame_start = int(frame_start)
        self.frame_end = int(frame_end)

    def duration(self) -> float:
        return self.time_end - self.time_start

    def num_frames(self) -> int:
        return self.frame_end - self.frame_start + 1

    def set_hist(self, hist: np.ndarray, loc: tuple):
        self.histograms[loc[0] * 3 + loc[1]] = hist

    def get_hist(self, loc: tuple) -> np.ndarray:
        return self.histograms[loc[0] * 3 + loc[1]]

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "<Segment[{:s}, start={:.1f} sec, end={:.1f} sec, duration={:.1f} sec, num_frames={:d}]>".format(self.movie_id, self.time_start, self.time_end, self.duration(), self.num_frames())

    def __hash__(self):
        return hash((self.movie_id, self.time_start, self.time_end, self.frame_start, self.frame_end))

    def __eq__(self, other):
        if not isinstance(other, self.__class__): return NotImplemented

        return self.movie_id == other.movie_id and self.time_start == other.time_start and self.time_end == other.time_end and self.frame_start == other.frame_start and self.frame_end == other.frame_end
