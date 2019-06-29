from src.Segment import Segment


class Video:
    segments: list
    name: str
    num_frames: int
    frame_rate: float

    def __init__(self, name: str, num_frames: int, frame_rate: float):
        self.name = name
        self.num_frames = num_frames
        self.frame_rate = frame_rate

    def add_segment(self, s: Segment) -> None:
        self.segments.append(s)

    def duration(self) -> float:
        return self.num_frames / self.frame_rate
        
    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "<Video[{:s}, {:.2f} sec, {:d} segments, {:d} frames, {:.2f} fps]>".format(self.name, self.duration(), len(self.segments), self.num_frames, self.frame_rate)
