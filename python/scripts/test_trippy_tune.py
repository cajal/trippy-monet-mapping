import random
import trippytune
from stimulus import stimulus

# verify against a random stimulus generated in experiments
key = random.choice(
    (stimulus.Trippy & (stimulus.Trial & 'animal_id in (20505, 20322, 20457, 20210, 20892)')).fetch('KEY'))
cond = (stimulus.Trippy * stimulus.Condition & key).out('movie').fetch1()

trippy = trippytune.Trippy.from_condition(cond)

assert 0 == abs(trippy.movie - (stimulus.Trippy & key).fetch1('movie')).max(),\
    "Python implementation diverged from MATLAB"
