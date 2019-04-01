import datajoint as dj
import numpy as np
import warnings
import os
import json
from stimulus import stimulus
from pipeline import fuse
from tqdm import tqdm

import monet_trippy as mt

# sessions that have both Monet and Trippy from a few recent experiments
sessions = (fuse.Activity * stimulus.Sync & 'animal_id in (20505, 20322, 20457, 20210, 20892)'
            & (stimulus.Trial * stimulus.Monet2) & (stimulus.Trial * stimulus.Trippy)).fetch('KEY')
key = sessions[0]   # pick one

print('load frame times.')
pipe = (fuse.Activity() & key).module
num_frames = (pipe.ScanInfo() & key).fetch1('nframes')
num_depths = len(dj.U('z') & (pipe.ScanInfo.Field().proj('z', nomatch='field') & key))
frame_times = (stimulus.Sync() & key).fetch1('frame_times', squeeze=True)  # one per depth
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

print('create session and load trials')
session = mt.VisualSession(np.stack(traces), frame_times)

print('load trippy trials')
for trial in tqdm((stimulus.Trial * stimulus.Condition * stimulus.Trippy & key).proj(..., '- movie')):
    try:
        session.add_trial(mt.Trippy.from_condition(trial), trial['flip_times'].flatten())
    except IndexError:
        warnings.warn('Invalid trial.')

print('load monet trials')
for trial in tqdm((stimulus.Trial * stimulus.Condition * stimulus.Monet2 & key)):
    try:
        session.add_trial(mt.Monet2.from_condition(trial), trial['flip_times'].flatten())
    except IndexError:
        warnings.warn('Invalid trial.')

folder = os.path.join(os.path.abspath('..'), 'data', 'sessions', dj.hash.key_hash(key)[:6])
print('save session', folder)

session.save(folder)
with open(os.path.join(folder, 'cajal.json')) as f:
    json.dump(key, f)
