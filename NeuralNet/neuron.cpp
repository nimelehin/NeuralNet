//
//  neuron.cpp
//  NeuralNet
//
//  Created by Никита Мелехин on 08.10.2018.
//  Copyright © 2018 Никита Мелехин. All rights reserved.
//

#include "neuron.hpp"
#include <string>
#include <vector>
#include <cmath>
#include <iostream>

using namespace std;

class neuron {
public:
    double outValue = 0.0;
    double inValue = 0.0;
    double error = 0.0;
    string activation = "sigmoid";
    vector<double>weightToNextLayer;
    neuron(int nextLayerSize) {
        weightToNextLayer.resize(nextLayerSize, 0.3);
        for (auto &i : weightToNextLayer){
            i = double(rand() % 1000 - 500) / double(1000);
        }
    }
    
    neuron(int nextLayerSize, string activationFunction) {
        this->activation = activationFunction;
        weightToNextLayer.resize(nextLayerSize, 0.3);
        for (auto &i : weightToNextLayer){
            i = double(rand() % 1000 - 500) / double(1000);
        }
    }
    
    neuron(int nextLayerSize, vector<double>w) {
        if (nextLayerSize == w.size()) {
            weightToNextLayer = w;
        } else {
            cout << nextLayerSize << " " << w.size();
            weightToNextLayer.resize(nextLayerSize, 0.3);
            for (auto &i : weightToNextLayer){
                i = double(rand() % 1000 - 500) / double(1000);
            }
        }
    }
    
    void setInValue(double _inValue) {
        this->inValue = _inValue;
    }
    
    void addToInValue(double plus) {
        this->inValue += plus;
    }
    
    void setOutValue(double _outValue) {
        this->outValue = _outValue;
    }
    
    void calcOutValue(){
        this->setOutValue(activationFunction(this->inValue));
    }
    
    double getOutValue(){
        return this->outValue;
    }
    
    double deactivation() {
        if (activation == "sigmoid") {
            double real = 1 / (1 + exp(this->outValue));
            return (1 - real) * real;
        }
        else if (activation == "tanh") {
            double real = tanh(this->outValue);
            return double(1 - real * real);
        }
        else if (activation == "relu") {
            if (this->outValue >= 0.0) {
                return 1.0;
            }
            return 0.0;
        }
        else {
            return 0;
        }
    }
    
    void setError(double _error){
        this->error = _error;
    }
    
    void addError(double _error){
        this->error += _error;
    }
    
    double getError(){
        return this->error;
    }
    
    double activationFunction(double val) {
        if (activation == "sigmoid") {
            return 1 / (1 + exp(-val));
        }
        else if (activation == "tanh") {
            return tanh(val);
        }
        else if (activation == "relu") {
            return max(0.0, val);
        }
        else {
            return val;
        }
    }
    
    double calcDelta() {
        return this->error * this->deactivation();
    }
    
    void pushToLayer(vector<neuron> &layer){
        for (int neuronId = 0; neuronId < layer.size() - 1; neuronId++) {
            layer[neuronId].addToInValue(this->outValue * weightToNextLayer[neuronId]);
        }
    }
    
    void updateWithPosition(vector<neuron> &layer, int to, double w_delta, double lr = 0.15) {
        
        for (int i = 0; i < layer.size(); i++) {
            layer[i].weightToNextLayer[to] -= w_delta * layer[i].getOutValue() * lr;
            layer[i].addError(layer[i].weightToNextLayer[to] * this->error);
        }
        
        this->setError(0.0);
    }
    
    void calcOutError(double target) {
        this->error = this->getOutValue() - target;
    }
    
    void printWeight() {
        for (auto i : weightToNextLayer){
            cout << i << " ";
        }
        cout << "\n\n";
    }
    
};
