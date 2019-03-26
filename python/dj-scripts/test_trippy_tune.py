"""
This test relies on connectivity to the Cajal datajoint server to use experimental data
"""

import numpy as np
from stimulus import stimulus
import monet_trippy as mt

print('Pick a random stimulus trial from recent experiments')
mice = 'animal_id in (20505, 20322, 20457, 20210, 20892)'
key = np.random.choice(
    (stimulus.Trippy & (stimulus.Trial & mice)).fetch('KEY'))
cond = (stimulus.Trippy * stimulus.Condition & key).proj(..., '- movie').fetch1()

print('Synthesize trippy stimulus movie')
trippy = mt.Trippy.from_condition(cond)

print('Load actual movie from the database')
movie_from_database = np.rollaxis((stimulus.Trippy & key).fetch1('movie'), 2)

print('compare movies')
assert 0 == abs(trippy.movie - movie_from_database).max(),  "Python diverged from MATLAB"

print('save movie as mp4')
trippy.save()