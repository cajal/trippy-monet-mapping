import numpy as np
import os
import pickle
import hashlib

from skimage import transform


from .visual import Visual


class VisualSession:
    """
    A collection of movie clips synchronized with a collection of neural signals
    """

    def __init__(self, traces: np.ndarray, times: np.ndarray, delays: np.ndarray):
        """
        :param traces: array of size (N, T) array where T = number of time samples, N = number of units
        :param times: array of size (T,) or with sample times of the traces
        :param delays: array of size (N,) with relative delays for each unit
        """
        assert times.ndim == 1, 'time must have one dimensions'
        assert traces.ndim == 2, 'traces must be a 2D matrix'
        if traces.ndim == 1:
            traces = traces[None, :]
        self.traces = traces
        self.start_time = times.min()
        self.times = times - self.start_time
        self.delays = delays
        self.trials = []

    def save(self, folder):
        os.makedirs(folder, exist_ok=True)

        trial_folder = os.path.join(folder, 'trials')
        stim_folder = os.path.join(folder, 'stims')
        os.path.isdir(trial_folder) or os.mkdir(trial_folder)
        os.path.isdir(stim_folder) or os.mkdir(stim_folder)

        # save traces and times
        np.savez(os.path.join(folder, 'traces.npz'), traces=self.traces, times=self.times, delays=self.delays)

        # save trials
        for trial in self.trials:
            stim_class = trial['stimulus'].__class__.__name__
            params = trial['stimulus'].params
            stim_hash = hashlib.md5(pickle.dumps(
                [v for k, v in sorted(params.items()) if not k.startswith('_')])).hexdigest()[:8]
            np.savez(os.path.join(trial_folder, 'trial-%08.3f.npz' % trial['times'][0]),
                     times=trial['times'],
                     stim_class=stim_class,
                     stim_hash=stim_hash)
            np.savez(os.path.join(stim_folder, 'stim-{stim_class}-{stim_hash}'.format(
                stim_class=stim_class, stim_hash=stim_hash)), params)

    def add_trial(self, stimulus: Visual, frame_times: np.ndarray):
        """
        :param stimulus: an object of type StimulusMovie
        :param frame_times: (s) array of size (T,)
        """

        if stimulus.nframes < frame_times.size:
            frame_times = frame_times[stimulus.nframes:]
            assert np.diff(frame_times).max() < 0.04
        self.trials.append({'stimulus': stimulus, 'times': frame_times - self.start_time})

    def compute_receptive_field(self, shape=None, temp_band=4.0, latency=0.15):
        """
        :param shape: (ny, nx) image dimensions. If None, use movie dimensions from the first
        :param temp_band: (Hz) temporal bandwidth of signal interpolation
        :param latency: (s) the delay at which to measure neural signals since the stimulus.
        :param latency: (s) the delay at which to measure neural signals since the stimulus.
        :return: (N, ny, nx) np.ndarray of correlations
        """
        if not self.trials:
            raise TypeError('No trials. Use VisualSession.add_trial to add trials')

        if shape is None:
            shape = self.trials[0]['stimulus'].movie.shape[:2]

        for t in self.trials:
            return t
