#ifndef SYNAPSELIB_LIBRARY_H
#define SYNAPSELIB_LIBRARY_H

#include <vector>

/**
 * @brief A basic Perceptron (single-layer neuron).
 */
class Perceptron {
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

protected:
    static double sigmoid(double x);
    double weightedSum(const std::vector<double>& inputs) const;
    std::vector<double> mWeights;
};

/**
 * @brief A basic Neural Network class, it has chosen number of hidden layers and Perceptron in each layer.
 */
class NeuralNetwork {
public:
    // Constructor
};

#endif //SYNAPSELIB_LIBRARY_H