#include <algorithm>
#include <iostream>
#include <random>
#include <vector>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
#include <string.h>
#include <random>
// #include "cifar-10-cnn.cpp"
#include "test_image.h"
#include "cnn.h"
using namespace std;

int main() {
     int output_final[1];
    int ido=0;
    for(int k=0;k<1;k++){
    double flattenedImage[32*32*3];
    int idx = 0;
    for(int d=0;d<3;d++){
    for (int i = 0; i < 32; ++i) {
        for (int j = 0; j < 32; ++j) {
            flattenedImage[idx++] = test_image[ido++];
        }
    }
    }
        double output[10];  // Assuming the output size is 10
        CNN(flattenedImage, 32,32, output);
       auto maxElementIterator = std::max_element(output, output + 10);
        int maxIndex = std::distance(output, maxElementIterator);
        output_final[k]=maxIndex;
    }

    return 0;
}
