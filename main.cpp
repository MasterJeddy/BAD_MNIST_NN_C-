#include <SFML/Graphics.hpp>
#include <iostream>
#include "MNIST_EigenLoader.h"
#include "Neural_Network.h"

double increment(double a){
    return a+0.4;
}

int main()
{


    MNIST_EigenLoader loader;
    std::vector<LabeledImage*> trainingData = loader.loadTrainingData();
    std::vector<LabeledImage*> testingData = loader.loadTestingData();

    Neural_Network nn = Neural_Network({784,30,10});

    std::vector<NeuralNetwork_Data> formattedTrainingData;
    std::vector<NeuralNetwork_Data> formattedTestData;

    for (auto data : trainingData){
        Eigen::VectorXd input(784);
        for (int i =0;i<784;i++){
            input(i) = (double)data->image(i)/255.0;
        }
        Eigen::VectorXd  expectedOutput(10);
        expectedOutput.setZero();
        expectedOutput(data->label) =1;
        formattedTrainingData.push_back({input,expectedOutput});
    }

    for (auto data : testingData){
        Eigen::VectorXd input(784);
        for (int i =0;i<784;i++){
            input(i) = (double)data->image(i)/255.0;
        }
        Eigen::VectorXd  expectedOutput(10);
        expectedOutput.setZero();
        expectedOutput(data->label) =1;
        formattedTestData.push_back({input,expectedOutput});
    }

    nn.simpleGradientDescent(&formattedTrainingData,30,10,0.5,&formattedTestData);

//    sf::RenderWindow window(sf::VideoMode(1280, 720), "MNIST LOADED NUMBERS");
//
//
//    std::vector<sf::Sprite> numbers;
//    std::vector<sf::Texture> textures;
//    sf::Uint8* pixels = new sf::Uint8[28 * 28 * 4];
//
//    for (int k=0;k<45*25;k++){
//        numbers.emplace_back();
//        textures.emplace_back();
//        textures[k].create(28,28);
//        for (int i =0;i<28*28;i++){
//            pixels[i*4] = formattedTestData[k].input(i)*255;
//            pixels[i*4+1] = formattedTestData[k].input(i)*255;
//            pixels[i*4+2] = formattedTestData[k].input(i)*255;
//            pixels[i*4+3] = 255;
//        }
//        textures[k].update(pixels);
//    }
//
//    for (int i=0;i<45;i++){
//        for (int k =0;k<25;k++){
//            numbers[k*45+i].setTexture(textures[k*45+i]);
//            numbers[k*45+i].setPosition(i*28,k*28);
//        }
//    }
//
//
//    while (window.isOpen())
//    {
//        sf::Event event;
//        while (window.pollEvent(event))
//        {
//            if (event.type == sf::Event::Closed)
//                window.close();
//        }
//
//        window.clear();
//
//        for (auto& num:numbers){
//            window.draw(num);
//        }
//
//        window.display();
//    }

    return 0;
}