import random
import trippytune
from stimulus import stimulus

# verify against some stimuli generated in experiments

cond_key = random.choice(
    (stimulus.Trippy & (stimulus.Trial & 'animal_id in (20505, 20322, 20457, 20210, 20892)')).fetch('KEY'))
cond = (stimulus.Trippy & cond_key).fetch1()

trippy = trippytune.Trippy.from_condition(cond)

assert 0 == abs(trippy.movie - cond['movie']).max(), "Python implementation diverged from MATLAB"