"""
create a trippy stimulus with default settings and save it into ./Trippy.mp4
"""
import monet_trippy as mt

trippy = mt.Trippy()
trippy.export(quality=7.5)
