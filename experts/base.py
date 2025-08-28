class Expert:
    def __init__(self, name):
        self.name = name

    def predict(self, data):
        "returns buy sell hold"
        raise NotImplementedError("This method should be overridden by subclasses")
    
    def __repr__(self):
        return f"Expert({self.name})"