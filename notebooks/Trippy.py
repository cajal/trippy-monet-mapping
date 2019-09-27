import datajoint as dj
import numpy as np
from stimulus import stimulus
from pipeline import reso, experiment, fuse
from stimline import tune
from scipy.interpolate import InterpolatedUnivariateSpline


class SplineCurve:
    def __init__(self, t, s, **kwargs):
        t0 = np.median(t)  # center for numerical stability
        self.t0 = t0
        if len(t.shape) == 1:
            self.splines = [InterpolatedUnivariateSpline(t - t0, si, **kwargs) for si in s]
        else:
            self.splines = [InterpolatedUnivariateSpline(ti - t0, si, **kwargs) for ti, si in zip(t, s)]

    def __len__(self):
        return len(self.splines)

    def __call__(self, t):
        return np.vstack([s(t - self.t0) for s in self.splines])


m = 'animal_id in (20505, 20322, 20457, 20210, 20892)'

keys = (fuse.Activity() * stimulus.Sync * tune.StimulusType
        & stimulus.Trial * stimulus.Condition & m & 'stimulus_type = "stimulus.Trippy"').fetch('KEY')

key = keys[0]

pipe = (fuse.Activity() & key).module
num_frames = (pipe.ScanInfo() & key).fetch1('nframes')
num_depths = len(dj.U('z') & (pipe.ScanInfo.Field().proj('z', nomatch='field') & key))

frame_times = (stimulus.Sync() & key).fetch1('frame_times', squeeze=True) # one per depth
assert num_frames <= frame_times.size / num_depths <= num_frames + 1
frame_times = frame_times[:num_depths * num_frames:num_depths] # one per volume

units = pipe.ScanSet.Unit * pipe.MaskClassification.Type & {'type': 'soma'}
spikes = pipe.Activity.Trace * pipe.ScanSet.UnitInfo & units & key

trace_keys, traces, ms_delay = spikes.fetch('KEY', 'trace', 'ms_delay')

print('Done')