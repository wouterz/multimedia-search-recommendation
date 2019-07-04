from .preprocess import load_training_set, get_test_video_set
from .evaluation import pick_test_segments, generate_test_segments, evaluate_segments, evaluate
from .Video import Video
from .Segment import Segment
import ipywidgets as widgets


def test_inspect_widgets(test_index):
    test_set_id_widget  = widgets.Dropdown(options=[('-', '-')] + [("Test set %d" % (i+1), i) for i in range(len(test_index))], description='Test set:', disabled=False)
    test_set_vid_widget = widgets.Dropdown(options=' - ', description='Video:', disabled=True)

    def change_x(args):
        if args['new'] == '-':
            test_set_vid_widget.disabled = True
            test_set_vid_widget.options = []
        else:
            test_set_vid_widget.disabled = False
            test_set_vid_widget.options = [('{:s} ({:d} to {:d})'.format(*x), x) for x in test_index[int(args['new'])]]

    test_set_id_widget.observe(change_x, 'value')

    return test_set_id_widget, test_set_vid_widget


def chunks(l, n):
    """Yield successive n-sized chunks from l."""
    for i in range(0, len(l), n):
        yield l[i:i + n]