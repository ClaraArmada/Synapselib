#ifndef SYNAPSELIB_LIBRARY_H
#define SYNAPSELIB_LIBRARY_H

#include <vector>

/**
 * @brief A basic Perceptron (single-layer neuron).
 */
class Perceptron {
    std::vector<double> mWeights;
public:
    // Constructor
    explicit Perceptron(const std::vector<double>& weights);

    // Change the weight at a specific index
    void weightChange(double newWeight, int index);

    // Compute the output for a given input
    double step(const std::vector<double>& inputs) const;

    // Train perceptron with input data
    void training(const std::vector<double>& inputs, double expectedOutput,
                  double learningRate = 0.05, int maxIterations = 1000,
                  double destinationErrorRate = 1e-6);

    // Get the current weights
    std::vector<double> getWeights() const;

    // Performs a sigmoid function on x (double)
    static double sigmoid(double x);

protected:
    double weightedSum(const std::vector<double>& inputs) const;
};

/**
* @brief Initializes a feedforward neural network with randomly assigned weights for each perceptron (neuron).
* The user specifies the number of neurons in the Input Layer, the structure of hidden layers, and the number of neurons in the output layer.
 */
class NeuralNetwork {
    int inputLayer;
    std::vector<std::vector<Perceptron>> hiddenLayer;
    std::vector<Perceptron> outputLayer;
public:
    // Constructor
    explicit NeuralNetwork(int inputLayerLength, std::vector<int> hiddenLayersLengths,
                           int outputLayerLength, std::vector<double> initialWeightsRange);

    // Single epoch, computes the output(s) of (an) input(s)
    double epoch(const std::vector<double>& inputs) const;

    // Performs a forward pass
    std::tuple<std::vector<double>, std::vector<double>, std::vector<double>>
        forwardPass(std::vector<double> inputs);

    //performs a loss calculation
    double lossCalculation(std::vector<double> expectedValues,
                            std::vector<double>);

    void backPropagation(std::vector<double> expectedValues, std::vector<double> inputs,
                         std::vector<double> outputActivations, std::vector<double> activations,
                         std::vector<double> weightedSum);


};

#endif //SYNAPSELIB_LIBRARY_H