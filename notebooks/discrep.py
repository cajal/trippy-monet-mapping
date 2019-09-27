import datajoint as dj
from stimulus import stimulus

schema = dj.schema('dimitri_debug')
@schema 
class Discrepancy(dj.Computed):
    definition = """
    -> stimulus.Trial
    ---
    discrepancy : float   # difference between recorded vs displayed frames
    """
    
    key_source = stimulus.Trial & (stimulus.Condition * stimulus.Trippy) & 'animal_id>16000'
    
    
    def make(self, key):
        duration, fps, times = (stimulus.Trippy * stimulus.Trial & key).fetch1('duration', 'fps', 'flip_times', squeeze=True)
        self.insert1(dict(key, discrepancy=duration*fps - times.size))

        
Discrepancy.populate(display_progress=True)
