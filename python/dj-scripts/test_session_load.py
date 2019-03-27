import datajoint as dj
import numpy as np
import os
from stimulus import stimulus
from pipeline import fuse
from stimline import tune

import monet_trippy as mt

# Find sessions from the Monet-Trippy study

m = 'animal_id in (20505, 20322, 20457, 20210, 20892)'

keys = (fuse.Activity * stimulus.Sync * tune.StimulusType
        & stimulus.Trial * stimulus.Condition & m & 'stimulus_type = "stimulus.Monet2"').fetch('KEY')
key = keys[2]

# load from datajoint

pipe = (fuse.Activity() & key).module
num_frames = (pipe.ScanInfo() & key).fetch1('nframes')
num_depths = len(dj.U('z') & (pipe.ScanInfo.Field().proj('z', nomatch='field') & key))

frame_times = (stimulus.Sync() & key).fetch1('frame_times', squeeze=True) # one per depth
assert num_frames <= frame_times.size / num_depths <= num_frames + 1
frame_times = frame_times[:num_depths * num_frames:num_depths]  # one per volume

# load and cache traces
trace_hash = dj.hash.key_hash({k: v for k, v in key.items() if k not in {'stimulus_type'}})
archive = os.path.join('cache', trace_hash + '-traces.npz')
if os.path.isfile(archive):
    data = np.load(archive)
    trace_keys = data['trace_keys']
    traces = data['traces']
    ms_delay = data['ms_delay']
else:
    units = pipe.ScanSet.Unit * pipe.MaskClassification.Type & {'type': 'soma'}
    spikes = pipe.Activity.Trace * pipe.ScanSet.UnitInfo & units & key
    trace_keys, traces, ms_delay = spikes.fetch('KEY', 'trace', 'ms_delay')
    np.savez_compressed(archive, trace_keys=trace_keys, traces=traces, ms_delay=ms_delay)
frame_times = np.add.outer(ms_delay / 1000, frame_times)  # num_traces x num_frames

session = mt.VisualSession(np.stack(traces), frame_times)

print('Done')
