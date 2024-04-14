//
// Created by simon on 1/29/24.
//

#include "Neural_Network.h"
#include <random>
#include <algorithm>
#include <iostream>
#include <chrono>

Neural_Network::Neural_Network(const std::vector<int>& sizes) {
    this->numLayers = sizes.size();
    this->sizes = sizes;
    std::default_random_engine generator;
    generator.seed(std::chrono::system_clock::now().time_since_epoch().count());
    std::normal_distribution<double> distribution(0.0,1.0);
    for (int i =1;i<numLayers;i++){
        biases.emplace_back(sizes[i]);
        for (int k =0;k<sizes[i];k++){
            biases[i-1](k) = distribution(generator);
        }
    }

    for (int i =0;i<numLayers-1;i++){
        distribution = std::normal_distribution(0.0,1/sqrt(sizes[i]));
        weights.emplace_back(sizes[i+1],sizes[i]);
        for (int k=0;k<sizes[i+1];k++){
            for (int j =0;j<sizes[i];j++){
                weights[i](k,j) = distribution(generator);
            }
        }
    }

}

Eigen::VectorXd Neural_Network::feedforward(const Eigen::VectorXd& input) {
    if (input.size() != sizes[0])
        throw "Input size does not match network";

    Eigen::VectorXd* activations = new Eigen::VectorXd[numLayers];

    for (int i =0;i<numLayers;i++){
        activations[i] = Eigen::VectorXd(sizes[i]);
    }
    activations[0] = input;

    for (int i = 1;i<numLayers;i++){
        activations[i] = (weights[i-1]*activations[i-1] +biases[i-1]).unaryExpr(std::ref(sigmoid));
    }

    Eigen::VectorXd output = activations[numLayers-1];

    delete[] activations;
    return output;
}

void Neural_Network::simpleGradientDescent(std::vector<NeuralNetwork_Data> *trainingData, int epochs, int miniBatchSize,
                                           double trainingRate, std::vector<NeuralNetwork_Data> *testData) {
    if (!trainingData)
        throw "No training data provided";

    std::default_random_engine generator;
    generator.seed(std::chrono::system_clock::now().time_since_epoch().count());


    for (int i = 0;i<epochs;i++){
        std::shuffle(std::begin(*trainingData),std::end(*trainingData),generator);
        int total = 0;
        while (total<trainingData->size()){
            std::vector<NeuralNetwork_Data> miniBatch;
            for (int k=0;k<miniBatchSize && total<trainingData->size();k++){
                miniBatch.push_back((*trainingData)[total]);
                total++;
            }
            updateMiniBatch(miniBatch,trainingRate);
            miniBatch.clear();
        }

        if (testData){
            std::cout << "Epoch " << i << " " << testNetwork(*testData) << "/" << testData->size() << std::endl;

        } else {
            std::cout << "Epoch " << i << " complete" << std::endl;
        }
    }
}

void Neural_Network::updateMiniBatch(std::vector<NeuralNetwork_Data> &miniBatch, double trainingRate) {
    std::vector<Eigen::VectorXd> biasDerivatives;
    for (auto & bias : biases){
        biasDerivatives.emplace_back(bias.rows());
    }
    std::vector<Eigen::MatrixXd> weightDerivatives;
    for (auto & weight : weights){
        weightDerivatives.emplace_back(weight.rows(),weight.cols());
    }

    for (int i =0;i<biasDerivatives.size();i++){
        biasDerivatives[i].setZero();
    }
    for (int i=0;i<weightDerivatives.size();i++){
        weightDerivatives[i].setZero();
    }


    for (auto& data:miniBatch){
        std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::MatrixXd>> derivativeBasedOnExample = backProp(data.input,data.expectedOutput);
        for (int i=0;i<biasDerivatives.size();i++){
            biasDerivatives[i] = biasDerivatives[i]+derivativeBasedOnExample.first[i];
        }
        for (int i=0;i<weightDerivatives.size();i++){
            weightDerivatives[i] = weightDerivatives[i] + derivativeBasedOnExample.second[i];
        }
    }

  //  std::cout << biases[1] << std::endl << std::endl;
    for (int i =0;i<biases.size();i++){
        biases[i] = biases[i] - (trainingRate/(double)miniBatch.size())*biasDerivatives[i];
    }
  //  std::cout << biases[1]<< std::endl << std::endl;;

    for (int i =0;i<weights.size();i++){
        //Added term for regulrisation
        double reg = 1-(trainingRate*5)/50000;

        weights[i] = reg*weights[i]- (trainingRate/(double)miniBatch.size())*weightDerivatives[i];
    }

}

std::pair<std::vector<Eigen::VectorXd>, std::vector<Eigen::MatrixXd>>
Neural_Network::backProp(const Eigen::VectorXd& input, const Eigen::VectorXd& expectedOutput) {


    Eigen::VectorXd activation = input;

    Eigen::VectorXd* activations = new Eigen::VectorXd[numLayers];
    for (int i =0;i<numLayers;i++){
        activations[i] = Eigen::VectorXd(sizes[i]);
    }
    Eigen::VectorXd* unweightedInputs = new Eigen::VectorXd[numLayers-1];
    for (int i =0;i<numLayers-1;i++){
        unweightedInputs[i] = Eigen::VectorXd(sizes[i+1]);
    }

    activations[0] = activation;


    for (int i = 0;i<numLayers-1;i++){
        unweightedInputs[i] = (weights[i]*activation +biases[i]);
        activations[i+1] = unweightedInputs[i].unaryExpr(std::ref(sigmoid));
        activation = activations[i+1];
    }

    Eigen::VectorXd error;

    //Qauratic
    //error = costDerivative(activations[numLayers-1],expectedOutput).array()*unweightedInputs[numLayers-2].unaryExpr(std::ref(sigmoidPrime)).array();
    //Cross Entropy
    error = (activations[numLayers-1]-expectedOutput);

    std::vector<Eigen::VectorXd> biasDerivatives;
    for (auto & bias : biases){
        biasDerivatives.emplace_back(bias.rows());
    }
    std::vector<Eigen::MatrixXd> weightDerivatives;
    for (auto & weight : weights){
        weightDerivatives.emplace_back(weight.rows(),weight.cols());
    }

    biasDerivatives[biasDerivatives.size()-1] =error;
    weightDerivatives[weightDerivatives.size()-1]=error*activations[numLayers-2].transpose();

    for (int layer = numLayers-3;layer>=0;layer--){
        error = (weights[layer+1].transpose()*error).array()*unweightedInputs[layer].unaryExpr(std::ref(sigmoidPrime)).array();
        biasDerivatives[layer] =error;
        weightDerivatives[layer]= error*activations[layer].transpose();
    }



    delete[] activations;
    delete[] unweightedInputs;

    return {biasDerivatives,weightDerivatives};
}

int Neural_Network::testNetwork(std::vector<NeuralNetwork_Data> &testingData) {
    int successes = 0;
    for (auto & i : testingData){
        Eigen::VectorXd outputActivations = feedforward(i.input);

        int index = 0;
        for (int k = 0;k<outputActivations.rows();k++){
            if (outputActivations(k) > outputActivations(index)){
                index = k;
            }
        }


        if (i.expectedOutput(index) == 1){
            successes++;
        }

    }
    return successes;
}

Eigen::VectorXd Neural_Network::costDerivative(const Eigen::VectorXd& outputActivations, const Eigen::VectorXd& expectedOutput) {
    return outputActivations-expectedOutput;
}

double Neural_Network::sigmoid(double input) {
    return 1.0/(1+std::exp(-input));
}

double Neural_Network::sigmoidPrime(double input) {
    return sigmoid(input)*(1.0-sigmoid(input));
}
