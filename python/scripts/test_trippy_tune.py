import trippytune
from stimulus import stimulus

cond_key = (stimulus.Trippy &
            (stimulus.Trial & 'animal_id in (20505, 20322, 20457, 20210, 20892)')).head(limit=1, as_dict=True)[0]

trippy = trippytune.Trippy(
    **{k: v for k, v in cond_key.items() if k in {
        'fps', 'rng_seed', 'packed_phase_movie', 'up_factor', 'temp_freq',
        'temp_kernel_length', 'duration', 'spatial_freq'}},
    tex_size=(cond_key['tex_xdim'], cond_key['tex_ydim']),
    nodes=(cond_key['xnodes'], cond_key['ynodes']))

m = trippy.compute_phase_movie()

print(type(m))