#include "library.h"

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

namespace py = pybind11;

PYBIND11_MODULE(Synapselib, m) {
    m.doc() = "Python bindings for Clara's neural network library";

    // Enum binding
    py::enum_<e_ActivationFunctions>(m, "ActivationFunction")
        .value("Sigmoid", e_ActivationFunctions::Sigmoid)
        .value("BinaryStep", e_ActivationFunctions::BinaryStep)
        .value("Linear", e_ActivationFunctions::Linear)
        .value("Tanh", e_ActivationFunctions::Tanh)
        .value("ReLU", e_ActivationFunctions::ReLU)
        .value("LeakyReLU", e_ActivationFunctions::LeakyReLU)
        .value("ParametricRelu", e_ActivationFunctions::ParametricRelu)
        .value("ELU", e_ActivationFunctions::ELU)
        .export_values();

    // Perceptron
    py::class_<Perceptron>(m, "Perceptron")
        .def(py::init<const std::vector<double>&, double>())
        .def("get_weights", &Perceptron::getWeights)
        .def("access_weights", &Perceptron::accessWeights)
        .def("get_bias", &Perceptron::getBias)
        .def("weight_change", &Perceptron::weightChange);

    // ActivatedPerceptron
    py::class_<ActivatedPerceptron>(m, "ActivatedPerceptron")
        .def(py::init<const std::vector<double>&, double, e_ActivationFunctions, double>())
        .def("weighted_sum", &ActivatedPerceptron::weightedSum)
        .def("step", &ActivatedPerceptron::step)
        .def("train", &ActivatedPerceptron::training);

    // NeuralNetwork
    py::class_<NeuralNetwork>(m, "NeuralNetwork")
        .def(py::init<>())  // default constructor
        .def(py::init<int,
                      const std::vector<std::pair<int, e_ActivationFunctions>>&,
                      std::pair<int, e_ActivationFunctions>,
                      const std::vector<double>&,
                      double>(),
             py::arg("input_layer_length"),
             py::arg("hidden_layers"),
             py::arg("output_layer"),
             py::arg("initial_weight_range"),
             py::arg("bias") = 0.0)
        .def("predict", &NeuralNetwork::predict)
        .def("forward_pass", &NeuralNetwork::forwardPass)
        .def("train", &NeuralNetwork::training);

    // ConvolutionLayer
    py::class_<ConvolutionLayer>(m, "ConvolutionLayer")
        .def(py::init<const std::vector<int>&, int, int>(),
             py::arg("kernelsFromCenter"),
             py::arg("kernelsCount"),
             py::arg("channelCount"))
        .def("convolution", &ConvolutionLayer::convolution)
        .def("update_weights", &ConvolutionLayer::updateWeights);

    // ConvolutionalNeuralNetwork
    py::class_<ConvolutionalNeuralNetwork>(m, "ConvolutionalNeuralNetwork")
        .def(py::init<>())  // default constructor
        .def(py::init<const std::vector<std::vector<ConvolutionLayer>>&,
                      const NeuralNetwork&>(),
             py::arg("convolutionBlocks"),
             py::arg("classifier"))
        .def("forward", &ConvolutionalNeuralNetwork::forward)
        .def("loss_calculation", &ConvolutionalNeuralNetwork::lossCalculation)
        .def("loss_derivative", &ConvolutionalNeuralNetwork::lossDerivative)
        .def("back_propagation", &ConvolutionalNeuralNetwork::backPropagation)
        .def("weight_updates", &ConvolutionalNeuralNetwork::weightUpdates)
        .def("train", &ConvolutionalNeuralNetwork::Training);
}