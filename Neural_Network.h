//
// Created by simon on 1/29/24.
//

#ifndef UNTITLED1_NEURAL_NETWORK_H
#define UNTITLED1_NEURAL_NETWORK_H

#include <Eigen/Dense>

struct NeuralNetwork_Data{
    Eigen::VectorXd input;
    Eigen::VectorXd expectedOutput;
};

class Neural_Network {
public:
    int numLayers;
    std::vector<int> sizes;
    std::vector<Eigen::VectorXd> biases;
    std::vector<Eigen::MatrixXd> weights;
public:
    Neural_Network(const std::vector<int>& sizes); //Done
    Eigen::VectorXd feedforward(const Eigen::VectorXd& input); //Done
    void simpleGradientDescent(std::vector<NeuralNetwork_Data> *trainingData, int epochs, int miniBatchSize, double trainingRate,
                               std::vector<NeuralNetwork_Data> *testData = nullptr); //Done
public:
    void updateMiniBatch(std::vector<NeuralNetwork_Data> &miniBatch, double trainingRate); //Done
    std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::MatrixXd>> backProp(const Eigen::VectorXd& input, const Eigen::VectorXd& expectedOutput);//Done
    int testNetwork(std::vector<NeuralNetwork_Data> &testingData); //Done
    Eigen::VectorXd costDerivative(const Eigen::VectorXd& outputActivations,const Eigen::VectorXd& expectedOutput); //Done
    static double sigmoid(double input); //Done
    static double sigmoidPrime(double input); //Done
};


#endif //UNTITLED1_NEURAL_NETWORK_H
