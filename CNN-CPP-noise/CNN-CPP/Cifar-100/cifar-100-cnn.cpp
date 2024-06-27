// #include <cmath>
// #include "cifar100_weights.h"
// using namespace std;
#include "cnn.h"

double relu(double x)
{
    return (x < 0) ? 0 : x;
}

typedef ap_fixed<32,16> fixed_type;  // Adjust the bit-width as needed

fixed_type lfsr(fixed_type state) {
    bool bit = state & 1;
    state = (state >> 1) ^ (-(bit) & 0xB400); // XOR with a primitive polynomial
    return state;
}

// Function to generate normally distributed noise using Box-Muller transform
fixed_type generateNormalNoise() {
    fixed_type u1 = static_cast<fixed_type>(lfsr(0xACE1u)) / RAND_MAX;  // Uniform random numbers
    fixed_type u2 = static_cast<fixed_type>(lfsr(0xACE1u)) / RAND_MAX;

    double log_value = log(static_cast<double>(u1));
    double cos_value = cos(2.0 * M_PI * static_cast<double>(u2));

    fixed_type z0 = static_cast<fixed_type>(sqrt(-2.0 * log_value) * cos_value);  // Box-Muller transform

    return z0;
}

void addNoise(double input[], double stddev, int input_size) {
    fixed_type mean = 0.0;

    for (int i = 0; i < input_size; i++) {
        fixed_type noise = generateNormalNoise();

        // Apply noise with mean = 0 and standard deviation = stddev
        input[i] = static_cast<double>(input[i]) + static_cast<double>(mean) + static_cast<double>(stddev) * static_cast<double>(noise);
    }
}

void softmax(const double input[], int size, double output[]) {
     double max_val = input[0];
    for (int i = 1; i < size; ++i) {
        if (input[i] > max_val) {
            max_val = input[i];
        }
    }

    double sum_exp = 0.0;
    for (int i = 0; i < size; ++i) {
        sum_exp += std::exp(input[i] - max_val);
    }

    for (int i = 0; i < size; ++i) {
        output[i] = std::exp(input[i] - max_val) / sum_exp;
        }

}


void convolution(const double flattenedImage[], const double kernels[], double output[], int imageWidth, int imageHeight, int imageDepth, int k_h, int k_w, const double biases[], int numKernels , char a_f)
{
    int output_h = imageHeight - k_h + 1;
    int output_w = imageWidth - k_w + 1;
    // int output_d = k_d;

    for (int k = 0; k < numKernels; ++k)
    {
        for (int i = 0; i < output_h; ++i)
        {
            for (int j = 0; j < output_w; ++j)
            {

                output[k * output_h * output_w + i * output_w + j] = biases[k]; // Initialize output with bias for the current kernel

                for (int kd = 0; kd < imageDepth; ++kd)
                {
                    for (int ki = 0; ki < k_h; ++ki)
                    {
                        for (int kj = 0; kj < k_w; ++kj)
                        {
                            output[k * output_h * output_w + i * output_w + j] +=
                                flattenedImage[ (j+kj) + (i+ki)*imageWidth + kd*imageHeight*imageWidth]  * kernels[k * (k_h * k_w*imageDepth) + ki * k_w + kd*k_h*k_w+kj];
                        }
                    }
                }
                if(a_f=='R'){
                    output[k * output_h * output_w + i * output_w + j] =  relu(output[k * output_h * output_w + i * output_w + j]);
                }
            }
        }
    }
}

void maxPooling(const double image[], double output[], int imageWidth, int numChannels, int pool_size) {
    int i_h = (imageWidth);
    int i_w = i_h;
    int output_h = i_h / pool_size;
    int output_w = i_w / pool_size;

    for (int c = 0; c < numChannels; ++c) {
        for (int i = 0; i < output_h; ++i) {
            for (int j = 0; j < output_w; ++j) {
                double max_val = 0.0;
                for (int pi = 0; pi < pool_size;pi+=2) {
                    for (int pj = 0; pj < pool_size; pj+=2) {
                        max_val = fmax(max_val, image[c * (i_h * i_w) + (i * pool_size + pi) * i_w + (j * pool_size + pj)]);
                    }
                }
                output[c * (output_h * output_w) + i * output_w + j] = max_val;
            }
        }
    }
}


void fullyConnectedLayer(const double input[], double output[], const double weights[], const double bias[], int inputSize, int outputSize, char a_f) {
    for (int i = 0; i < outputSize; ++i) {
        output[i] = bias[i];
        for (int j = 0; j < inputSize; ++j) {
            output[i] += input[j] * weights[i * inputSize + j];
        }

        if (a_f=='R') {
            output[i] = (output[i] < 0) ? 0 : output[i];
        }
    }
}


void CNN(const double flattenedImage[], int imageWidth, int imageHeight, double output[]) {
    // Convolutional layer1
    int num_kernels1 = 58;
    int kernelWidth1 = 3;
    int kernelHeight1 = 3;
    int imagedepth1 = 3;
    char a_f = 'R';
    double conv1Output[30*30*58];
    convolution(flattenedImage, convolution1_weights, conv1Output, imageWidth, imageHeight, imagedepth1, kernelHeight1, kernelWidth1 , convolution1_bias,num_kernels1, a_f);
    addNoise(conv1Output,0.054321, 30*30*58);
       // Convolutional layer2
    int num_kernels2 = 96;
    int kernelWidth2 = 3;
    int kernelHeight2 = 3;
    int imagedepth2 = 58;
    // char a_f = 'R';
    double conv2Output[28*28*96];
    convolution(conv1Output, convolution2_weights, conv2Output, 30,30, imagedepth2, kernelHeight2, kernelWidth2 , convolution2_bias,num_kernels2, a_f);
    addNoise(conv2Output,0.089123, 28*28*96);
   // convulation 3
    double conv3Output[26*26*64];
    convolution(conv2Output, convolution3_weights, conv3Output,28,28,96, 3,3 , convolution3_bias,64,a_f);
    addNoise(conv3Output,0.001234, 26*26*64);
    // convulation 4
    double conv4Output[24*24*68];
    convolution(conv3Output, convolution4_weights, conv3Output,26,26,64, 3,3 , convolution4_bias,68,a_f);
    addNoise(conv4Output,0.078901, 24*24*68);
    // convolution 5
    double conv5Output[22*22*84];
    convolution(conv4Output, convolution5_weights, conv5Output,24,24,68, 3,3 , convolution5_bias,84,a_f);
    addNoise(conv5Output,0.067890, 22*22*84);
    // convolution 6
    double conv6Output[20*20*58];
    convolution(conv5Output, convolution6_weights, conv6Output,22,22,84, 3,3 , convolution6_bias,58,a_f);
    addNoise(conv6Output,0.032109,20*20*58);
    // convolution 7
    double conv7Output[18*18*86];
    convolution(conv6Output, convolution7_weights, conv7Output,20,20,58, 3,3 , convolution7_bias,86,a_f);
    addNoise(conv7Output,0.045678,18*18*86);
        // convolution 8
    double conv8Output[16*16*42];
    convolution(conv7Output, convolution8_weights, conv8Output,18,18,86, 3,3 , convolution8_bias,42,a_f);
    addNoise(conv8Output,0.098765,16*16*42);
            // convolution 9
    double conv9Output[14*14*54];
    convolution(conv8Output, convolution9_weights, conv9Output,16,16,42, 3,3 , convolution9_bias,54,a_f);
    addNoise(conv9Output,0.012345,14*14*54);
                // convolution 10
    double conv10Output[12*12*72];
    convolution(conv9Output, convolution10_weights, conv10Output,14,14,54, 3,3 , convolution10_bias,72,a_f);
    addNoise(conv10Output,0.076543,12*12*72);
    // fully connected layer. 
    char a_f2='S';
    double fully1output[100];
    fullyConnectedLayer(conv10Output,fully1output,dense1_weights,dense1_bias,10368,100,a_f2);
    char a_f3='S';
    if(a_f3=='S'){
        softmax(fully1output,100,output);
    }
}




