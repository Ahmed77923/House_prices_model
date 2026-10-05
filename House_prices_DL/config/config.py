class Modelconfig:
    def __init__(self):
        self.INPUT_DIM = 0
        self.hidden_dim = 128
        self.output_dim = 1
        self.num_layers = 2
        self.DROPOUT = 0.2
        
class Trainingconfig:
    def __init__(self):
        self.LR = 0.001
        self.BATCH_SIZE = 32
        self.NUM_EPOCHS = 100
        self.TEST_SIZE = 0.2
        self.RANDOM_STATE = 42
        
class Config:
    Modelconfig = Modelconfig()
    Trainingconfig = Trainingconfig()