import numpy as np

from .visual import Visual


class VisualSession:
    """
    A collection of movie clips synchronized with a collection of neural signals
    """

    def __init__(self, traces: np.ndarray, times: np.ndarray):
        """
        :param traces: array of size (T,) or (N, T) array where T = number of time samples, N = number of units
        :param times: array of size (T,) or (N, T) with sample times of the traces
        """
        if traces.ndim == 1:
            traces = traces[None, :]
        if times.ndim == 1:
            times = times[None, :]
        if traces.shape[1] != times.shape[1]:
            raise TypeError('The number of samples in the traces does not match the number of timestamps')
        self.traces = traces
        self.times = times
        self.trials = []

    def add_trial(self, stimulus_movie: Visual, frame_times: np.ndarray):
        """
        :param stimulus_movie: an object of type StimulusMovie
        :param frame_times: (s) array of size (T,)
        """
        if stimulus_movie.nframes != frame_times.size:
            raise IndexError('frame times must match stimulus movie')
        self.trials.append((stimulus_movie, frame_times))
