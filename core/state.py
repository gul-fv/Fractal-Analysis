class State:
    def __init__(self):
        self.model = None
        self.labels = None
        self.image_list = []
        self.image_count = 0
        self.source_type = None
        self.min_conf = 0.4
