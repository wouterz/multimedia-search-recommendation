from src.Segment import Segment


class Video:
    segments: list

    def add_segment(self, s: Segment) -> None:
        self.segments.append(s)
