"""
This test relies on connectivity to the Cajal datajoint server to use experimental data
"""

import random
from stimulus import stimulus
import trippytune

mice = 'animal_id in (20505, 20322, 20457, 20210, 20892)'

# verify against a random stimulus generated in experiments
key = random.choice(
    (stimulus.Trippy & (stimulus.Trial & mice)).fetch('KEY'))
cond = (stimulus.Trippy * stimulus.Condition & key).proj(..., '- movie').fetch1()

trippy = trippytune.Trippy.from_condition(cond)

assert 0 == abs(trippy.movie - (stimulus.Trippy & key).fetch1('movie')).max(),\
    "Python implementation diverged from MATLAB"
