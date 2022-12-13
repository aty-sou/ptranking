

class FederatedRanker:
    "An abstract ranker which is designed for federated optimization"

    def get_named_parameters(self):
        pass

    def get_parameter_names(self):
        pass

    def get_gradients(self):
        pass

    def get_named_gradients(self):
        pass

    def on_device_learning(self):
        pass

    def generate_SERPs(self):
        pass

    def update_ranker(self):
        pass

    def get_updated_weights(self):
        pass
