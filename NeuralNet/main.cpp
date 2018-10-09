//
//  main.cpp
//  NeuralNet
//
//  Created by Никита Мелехин on 08.10.2018.
//  Copyright © 2018 Никита Мелехин. All rights reserved.
//

#include <iostream>
#include "net.cpp"

int main(int argc, const char * argv[]) {
    
    net myNet(0.65, 500);
    
    vector<vector<double>>X, Y;
    
    
    vector<double>x1, y1;
    x1.push_back(1);
    x1.push_back(1);
    x1.push_back(0);
    y1.push_back(0);
    y1.push_back(0);
    X.push_back(x1);
    Y.push_back(y1);
    
    vector<double>x2, y2;
    x2.push_back(1);
    x2.push_back(0);
    x2.push_back(1);
    y2.push_back(1);
    y2.push_back(1);
    X.push_back(x2);
    Y.push_back(y2);
    
    vector<int>topology;
    topology.push_back(3);
    topology.push_back(10);
    topology.push_back(5);
    topology.push_back(4);
    topology.push_back(2);
    
    vector<vector<vector<double>>>ww;
    vector<vector<double>>layer1;
    vector<double>layer1_n1;
    layer1_n1.push_back(0.79);
    layer1_n1.push_back(0.85);
    layer1.push_back(layer1_n1);
    
    layer1_n1.clear();
    layer1_n1.push_back(0.44);
    layer1_n1.push_back(0.43);
    layer1.push_back(layer1_n1);
    
    layer1_n1.clear();
    layer1_n1.push_back(0.43);
    layer1_n1.push_back(0.29);
    layer1.push_back(layer1_n1);
    
    ww.push_back(layer1);
    layer1.clear();
    
    layer1_n1.clear();
    layer1_n1.push_back(0.5);
    layer1.push_back(layer1_n1);
    
    layer1_n1.clear();
    layer1_n1.push_back(0.52);
    layer1.push_back(layer1_n1);
    
    ww.push_back(layer1);
    ww.push_back(layer1);
    layer1.clear();
    
    myNet.createNet(topology);
    
    myNet.train(X, Y);
    
    auto res = myNet.feedForward(X[0]);
    
    for (auto i : res){
        cout << i << "\n";
    }
    
    res = myNet.feedForward(X[1]);
    
    for (auto i : res){
        cout << i << "\n";
    }
    
    return 0;
}
