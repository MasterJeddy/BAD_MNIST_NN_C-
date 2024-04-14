//
// Created by simon on 1/25/24.
//

#include "MNIST_EigenLoader.h"
#include <fstream>

MNIST_EigenLoader::MNIST_EigenLoader() = default;

std::vector<LabeledImage *> MNIST_EigenLoader::loadTrainingData() {
    uint32_t a;
    //Open file
    std::ifstream imagesFile( "Data/train-images.idx3-ubyte", std::ios::binary );
    std::ifstream labelFile("Data/train-labels.idx1-ubyte",std::ios::binary);
    //Loads and forgets magic number of images
    imagesFile.read ((char*)&a+3, 1);
    imagesFile.read ((char*)&a+2, 1);
    imagesFile.read ((char*)&a+1, 1);
    imagesFile.read ((char*)&a, 1);
    //Loads and forgets the amount of images
    imagesFile.read ((char*)&a+3, 1);
    imagesFile.read ((char*)&a+2, 1);
    imagesFile.read ((char*)&a+1, 1);
    imagesFile.read ((char*)&a, 1);
    //Loads and forgets Image height
    imagesFile.read ((char*)&a+3, 1);
    imagesFile.read ((char*)&a+2, 1);
    imagesFile.read ((char*)&a+1, 1);
    imagesFile.read ((char*)&a, 1);
    //Loads and forgets image width
    imagesFile.read ((char*)&a+3, 1);
    imagesFile.read ((char*)&a+2, 1);
    imagesFile.read ((char*)&a+1, 1);
    imagesFile.read ((char*)&a, 1);
    //Load and forget label magic number
    labelFile.read ((char*)&a+3, 1);
    labelFile.read ((char*)&a+2, 1);
    labelFile.read ((char*)&a+1, 1);
    labelFile.read ((char*)&a, 1);
    //Load and forget label count
    labelFile.read ((char*)&a+3, 1);
    labelFile.read ((char*)&a+2, 1);
    labelFile.read ((char*)&a+1, 1);
    labelFile.read ((char*)&a, 1);


    std::vector<LabeledImage*> out;


    //load images
    char c;
    for (int i = 0;i<50000;i++){
        out.push_back(new LabeledImage);
        labelFile.read ((char*)&c, 1);
        out[i]->label = (int)c;
        out[i]->image = Eigen::VectorXi(28*28);
        for (int k =0;k<28*28;k++){
            imagesFile.read ((char*)&c, 1);
            out[i]->image(k) = (int)c;
        }

    }

    imagesFile.close();
    labelFile.close();


    return out;
}

std::vector<LabeledImage *> MNIST_EigenLoader::loadTestingData() {
    uint32_t a;
    //Open file
    std::ifstream imagesFile( "Data/t10k-images.idx3-ubyte", std::ios::binary );
    std::ifstream labelFile("Data/t10k-labels.idx1-ubyte",std::ios::binary);
    //Loads and forgets magic number of images
    imagesFile.read ((char*)&a+3, 1);
    imagesFile.read ((char*)&a+2, 1);
    imagesFile.read ((char*)&a+1, 1);
    imagesFile.read ((char*)&a, 1);
    //Loads and forgets the amount of images
    imagesFile.read ((char*)&a+3, 1);
    imagesFile.read ((char*)&a+2, 1);
    imagesFile.read ((char*)&a+1, 1);
    imagesFile.read ((char*)&a, 1);
    //Loads and forgets Image height
    imagesFile.read ((char*)&a+3, 1);
    imagesFile.read ((char*)&a+2, 1);
    imagesFile.read ((char*)&a+1, 1);
    imagesFile.read ((char*)&a, 1);
    //Loads and forgets image width
    imagesFile.read ((char*)&a+3, 1);
    imagesFile.read ((char*)&a+2, 1);
    imagesFile.read ((char*)&a+1, 1);
    imagesFile.read ((char*)&a, 1);
    //Load and forget label magic number
    labelFile.read ((char*)&a+3, 1);
    labelFile.read ((char*)&a+2, 1);
    labelFile.read ((char*)&a+1, 1);
    labelFile.read ((char*)&a, 1);
    //Load and forget label count
    labelFile.read ((char*)&a+3, 1);
    labelFile.read ((char*)&a+2, 1);
    labelFile.read ((char*)&a+1, 1);
    labelFile.read ((char*)&a, 1);


    std::vector<LabeledImage*> out;


    //load images
    char c;
    for (int i = 0;i<10000;i++){
        out.push_back(new LabeledImage);
        labelFile.read ((char*)&c, 1);
        out[i]->label = (int)c;
        out[i]->image = Eigen::VectorXi(28*28);
        for (int k =0;k<28*28;k++){
            imagesFile.read ((char*)&c, 1);
            out[i]->image(k) = (int)c;
        }

    }

    imagesFile.close();
    labelFile.close();


    return out;
}
