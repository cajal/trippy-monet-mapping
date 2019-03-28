import datajoint as dj
import numpy as np
import os
from stimulus import stimulus
from pipeline import fuse

import monet_trippy as mt

# sessions that have both Monet and Trippy from a few recent experiments
sessions = (fuse.Activity * stimulus.Sync & 'animal_id in (20505, 20322, 20457, 20210, 20892)'
            & (stimulus.Trial * stimulus.Monet2) & (stimulus.Trial * stimulus.Trippy)).fetch('KEY')
key = sessions[2]   # pick one

print('load frame times.')
pipe = (fuse.Activity() & key).module
num_frames = (pipe.ScanInfo() & key).fetch1('nframes')
num_depths = len(dj.U('z') & (pipe.ScanInfo.Field().proj('z', nomatch='field') & key))
frame_times = (stimulus.Sync() & key).fetch1('frame_times', squeeze=True) # one per depth
assert num_frames <= frame_times.size / num_depths <= num_frames + 1
frame_times = frame_times[:num_depths * num_frames:num_depths]  # one per volume

print('load and cache soma traces')
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

print('create Trippy session and load trials')
trippy_session = mt.VisualSession(np.stack(traces), frame_times)
for trial in (stimulus.Trial * stimulus.Condition * stimulus.Trippy & key).proj(..., '- movie'):
    trippy_session.add_trial(mt.Trippy.from_condition(trial), trial['flip_times'].flatten())

print('compute receptive fields')
rf = trippy_session.compute_receptive_field()

print('Done')