from src.Segment import Segment


class Video:
    segments: list
    name: str

    def __init__(self, name: str):
        self.name = name

    def add_segment(self, s: Segment) -> None:
        self.segments.append(s)

    def __repr__(self):
        return self.__str__()

    def __str__(self):
        return "<Video[%s, %d segments]>" % (self.name, len(self.segments))
