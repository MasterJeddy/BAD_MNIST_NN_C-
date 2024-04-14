//
// Created by simon on 1/25/24.
//

#ifndef UNTITLED1_MNIST_EIGENLOADER_H
#define UNTITLED1_MNIST_EIGENLOADER_H

#include <Eigen/Dense>

struct LabeledImage{
    int label;
    Eigen::VectorXi image;
};

class MNIST_EigenLoader {
public:
    MNIST_EigenLoader();
    std::vector<LabeledImage*> loadTrainingData();
    std::vector<LabeledImage*> loadTestingData();
private:

};


#endif //UNTITLED1_MNIST_EIGENLOADER_H
