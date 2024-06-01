from .lstm_model import LSTMModel

class LSTMModelWrap():
    def __init__(self, input_size, hidden_layer_size, num_layers, output_size):
        self.model = LSTMModel(input_size, hidden_layer_size, num_layers, output_size)
