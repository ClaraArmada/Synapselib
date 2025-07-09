import importlib
import sys

spec = importlib.util.spec_from_file_location("synapselib", "cmake-build-release/synapselib.pyd")
Synapselib = importlib.util.module_from_spec(spec)
sys.modules["synapselib"] = Synapselib
spec.loader.exec_module(Synapselib)

nn = Synapselib.NeuralNetwork(
    3,
    [(4, Synapselib.ActivationFunction.Sigmoid)],
    (1, Synapselib.ActivationFunction.Sigmoid),
    [-1.0, 1.0]
)

nn.training([0.1, 0.2, 0.3], [1.0], 0.1, 1000, 100)
print(nn.predict([0.1, 0.2, 0.3]))