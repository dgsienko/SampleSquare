Class OneShot ():
    def __init__(self,json_row):
        self.id = json_row['_id']
        ableton_effects = json_row['ableton_effects_data']
        self.effects_names = ableton_effects[::2]
        self.effects_data = ableton_effects[1::2]