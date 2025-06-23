#include "library.h"

#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>

// Basic Neuron/Perceptron
class Perceptron {
    protected:
    // Sigmoid function
    static double sigmoid(const double x) {
        return 1.0f/(1.0f + exp(-x));
    }

    // Sum of weights * inputs
    double weightedSum(const std::vector<double>& inputs) const {
        return inner_product(inputs.begin(), inputs.end(), weights.begin(), 0.0);
    }
    std::vector<double> weights;

    public:
    // Constructor
    explicit Perceptron(const std::vector<double>& weights) {
        // Initial variables
        this->weights = weights;
        // float biases;
    }

    // Replaces previous weights with new inputed weights in the index chosen
    void weightChange(const double newWeights, const int index) {
        weights[index] = newWeights;
    }

    // A single step, does the sigmoid of the weighted sum without modifying weights
    double step(const std::vector<double>& inputs) const {
        return sigmoid(weightedSum(inputs));
    }

    // training of a perceptron, not to be used in neural networks (as we'll use backpropagation instead)
    void training(const std::vector<double>& inputs, const double expectedOutput, const double learningRate=0.05f, const int maxIterations=1000, double destinationError=1e-6) {
        for (int _ = 0; _ < maxIterations; _++) {

            const double output = step(inputs);

            const double error = expectedOutput - output;

            if (abs(error) < destinationError) {
                break;
            }

            for (int index = 0; index < weights.size(); index++) {
                const double gradient = error * output * (1 - output) * inputs[index];
                weights[index] += learningRate * gradient;
            }
        }
    }
};

int main() {
    // idk what to do here lolll
    return 0;
}