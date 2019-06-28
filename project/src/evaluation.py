from .Segment import Segment

def evaluate_segments(predicted: [Segment], labels: [Segment]):
    
    movie_correct = 0
    movie_wrong = 0

    for segment, label in zip(predicted, labels):

        # Check if movie is correct
        if segment == label: movie_correct += 1
        else: movie_wrong += 1

    total = movie_correct + movie_wrong
    fraction = movie_correct / total if total > 0 else 0

    print("Segment evaluation:")
    print("Correct: {:d}".format(movie_correct))
    print("Wrong:   {:d}".format(movie_wrong))
    print("Total:   {:d}".format(total))
    print("TPR:     {:.1f}%".format(fraction * 100), flush=True)