#include <algorithm>
#include <iostream>
#include <random>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <random>
#include <string.h>
// #include "mnist-cnn.cpp"
#include "test_image.h"
// #include "test_prediction.h"
#include "cnn.h"
using namespace std;

int main() {
     int output_final[1];
    int ido=0;
    for(int k=0;k<1;k++){
    double flattenedImage[28*28*1];
    int idx = 0;
    for(int d=0;d<1;d++){
    for (int i = 0; i < 28; ++i) {
        for (int j = 0; j < 28; ++j) {
            flattenedImage[idx++] = test_image[ido++];
        }
    }
    }
        double output[10];  // Assuming the output size is 10
        CNN(flattenedImage, 28,28, output);
       auto maxElementIterator = std::max_element(output, output + 10);
        int maxIndex = std::distance(output, maxElementIterator);
        output_final[k]=maxIndex;
    }
    // int count =0;
    // for(int i=0;i<100;i++){
    //     // cout<<output_final[i]<<" "<<test_prediction[i]<<endl;
    //     if(output_final[i]!=test_prediction[i]){
    //         count++;
    //     }
    // }
    // cout<<count<<endl;
    return 0;
}
