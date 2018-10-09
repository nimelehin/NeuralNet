//
//  net.cpp
//  NeuralNet
//
//  Created by Никита Мелехин on 08.10.2018.
//  Copyright © 2018 Никита Мелехин. All rights reserved.
//

#include "net.hpp"
#include "neuron.cpp"
#include <vector>
#include <iostream>

#define layer vector<neuron>

using namespace std;

class net {
    double learningRate = 0.1;
    int epoch = 5;
    vector<layer>neuralNet;
public:
    
    net(double _learningRate, int _epoch) {
        this->learningRate = _learningRate;
        this->epoch = _epoch;
    }
    
    ~net(){
        cout << "lel";
    }
    
    void createNet(vector<int>topology, vector<vector<vector<double>>>wew){
        for (int layerId = 0; layerId < topology.size(); layerId++) {
            int nextLayerSize = layerId == (topology.size() - 1) ? 0 : topology[layerId+1];
            int currentLayerSize = topology[layerId];
            vector<neuron>currentLayer;
            for (int neuronId = 0; neuronId < currentLayerSize; neuronId++) {
                neuron newNeuron(nextLayerSize, wew[layerId][neuronId]);
                currentLayer.push_back(newNeuron);
            }
            neuron biasNeuron(nextLayerSize);
            biasNeuron.setOutValue(1.0);
            currentLayer.push_back(biasNeuron);
            neuralNet.push_back(currentLayer);
        }
    }
    
    void createNet(vector<int>topology){
        for (int layerId = 0; layerId < topology.size(); layerId++) {
            int nextLayerSize = layerId == (topology.size() - 1) ? 0 : topology[layerId+1];
            int currentLayerSize = topology[layerId];
            vector<neuron>currentLayer;
            for (int neuronId = 0; neuronId < currentLayerSize; neuronId++) {
                neuron newNeuron(nextLayerSize);
                currentLayer.push_back(newNeuron);
            }
            neuron biasNeuron(nextLayerSize);
            biasNeuron.setOutValue(1.0);
            currentLayer.push_back(biasNeuron);
            neuralNet.push_back(currentLayer);
        }
    }
    
    vector<double> feedForward(vector<double>initValues) {
        for (int neuronId = 0; neuronId < neuralNet[0].size(); neuronId++) {
            neuralNet[0][neuronId].setOutValue(initValues[neuronId]);
        }
        
//        cout << "\n\n-----\nweights\n\n";
//        neuralNet[0][0].printWeight();
//        neuralNet[0][1].printWeight();
//        neuralNet[0][2].printWeight();
//        neuralNet[1][0].printWeight();
//        neuralNet[1][1].printWeight();
//        cout << "------\n\n\n";
        
        for (int layerId = 0; layerId < neuralNet.size() - 1; layerId++) {
            for (int neuronId = 0; neuronId < neuralNet[layerId+1].size(); neuronId++) {
                neuralNet[layerId+1][neuronId].setInValue(0.0);
            }
            
            for (int neuronId = 0; neuronId < neuralNet[layerId].size(); neuronId++) {
                neuralNet[layerId][neuronId].pushToLayer(neuralNet[layerId+1]);
            }
            
            for (int neuronId = 0; neuronId < neuralNet[layerId+1].size() - 1; neuronId++) {
                neuralNet[layerId+1][neuronId].calcOutValue();
            }
        }
        
        vector<double>result;
        
        for (int neuronId = 0; neuronId < neuralNet.back().size() - 1; neuronId++) {
            result.push_back(neuralNet.back()[neuronId].getOutValue());
        }
        
        return result;
    }
    
    void backProp (vector<double>X, vector<double>y) {
        feedForward(X);
        
        for (int neuronId = 0; neuronId < neuralNet.back().size() - 1; neuronId++) {
            neuralNet.back()[neuronId].calcOutError(y[neuronId]);
            double w_delta = neuralNet.back()[neuronId].calcDelta();
            //cout << "delta" << " " << w_delta << "\n\n";
            neuralNet.back()[neuronId].updateWithPosition(neuralNet[neuralNet.size() - 2], neuronId, w_delta, this->learningRate);
        }
        
        //neuralNet[1][0].printWeight();
        //neuralNet[1][1].printWeight();
        
        for (int layerId = neuralNet.size() - 2; layerId > 0; layerId--) {
            for (int neuronId = 0; neuronId < neuralNet[layerId].size() - 1; neuronId++) {
                double w_delta = neuralNet[layerId][neuronId].calcDelta();
                neuralNet[layerId][neuronId].updateWithPosition(neuralNet[layerId - 1], neuronId, w_delta, this->learningRate);
            }
        }
        
        //cout << "\n";
        
    }
    
    void train(vector<vector<double>>X, vector<vector<double>>y) {
        if (X.size() != y.size())return;
        
        for (int e = 0; e < this->epoch; e++){
            for (int i = 0; i < X.size(); i++){
                backProp(X[i], y[i]);
            }
        }
        
    }
    
};
