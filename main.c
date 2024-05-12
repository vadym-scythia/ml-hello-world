#include "stdio.h"
#include "main.h"
#include "stdlib.h"
#include "time.h"
#include "math.h"

int main() {
    neural_net *net = initialize_network(3, 2);

    float input[3] = {0.5, 0.3, 0.2};
    float output[2];

    forward_propagate_with_activation(net, input, output);

    printf("Output after softmax:\n");
    for (int i = 0; i < 2; i++) {
        printf("Output node %d: %f\n", i, output[i]);
    }
    printf("Sum should be equal to 1: %f + %f = %f", output[0], output[1], output[0] + output[1]);

    free_network(net);
    
    return 0;
}

int reverse_int(int i) {
    unsigned char c1, c2, c3, c4;
    c1 = i & 255;
    c2 = (i >> 8) & 255;
    c3 = (i >> 16) & 255;
    c4 = (i >> 24) & 255;
    return ((int)c1 << 24) + ((int)c2 << 16) + ((int)c3 << 8) + c4;
}

mnist_data* load_mnist_images(const char* images_filename) {
    FILE *file = fopen(images_filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Error during file opening %s\n", images_filename);
        return NULL;
    }

    int magic_number = 0, num_images = 0, num_rows = 0, num_cols = 0;
    fread(&magic_number, sizeof(magic_number), 1, file);
    magic_number = reverse_int(magic_number);
    if (magic_number != 2051) {
        fprintf(stderr, "Invalid MNIST images' file!\n");
        fclose(file);
        return NULL;
    }

    fread(&num_images, sizeof(num_images), 1, file);
    num_images = reverse_int(num_images);
    fread(&num_rows, sizeof(num_rows), 1, file);
    num_rows = reverse_int(num_rows);
    fread(&num_cols, sizeof(num_cols), 1, file);
    num_cols = reverse_int(num_cols);

    mnist_data *data = (mnist_data *)malloc(sizeof(mnist_data));
    data->size = num_images;
    data->images = (float **)malloc(num_images * sizeof (float *));

    for (int i = 0; i < num_images; i++) {
        data->images[i] = (float *)malloc(num_rows * num_cols * sizeof(float));
        for (int r = 0; r < num_rows; r++) {
            for (int c = 0; c < num_cols; c++) {
                unsigned char temp = 0;
                fread(&temp, sizeof(temp), 1, file);
                data->images[i][(num_rows * r) + c] = temp / 255.0f; // Normalize pixel values to [0, 1]
            }
        }
    }

    fclose(file);
    return data;
}

mnist_data* load_mnist_labels(const char* labels_filename) {
    FILE *file = fopen(labels_filename, "rb");
    if (file == NULL) {
        fprintf(stderr, "Cannot open file %s\n", labels_filename);
        return NULL;
    }

    int magic_number = 0, num_labels = 0;
    fread(&magic_number, sizeof(magic_number), 1, file);
    magic_number = reverse_int(magic_number);
    if (magic_number != 2049) {
        fprintf(stderr, "Invalid MNIST labels' file!");
        return NULL;
    }

    fread(&num_labels, sizeof(num_labels), 1, file);
    num_labels = reverse_int(num_labels);

    mnist_data *data = (mnist_data *)malloc(sizeof(mnist_data));
    data->size = num_labels;
    data->labels = (int *)malloc(num_labels * sizeof(int));

    for(int i = 0; i < num_labels; i++) {
        unsigned char temp = 0;
        fread(&temp, sizeof(temp), 1, file);
        data->labels[i] = (int)temp;
    }

    fclose(file);
    return data;
}

neural_net* initialize_network(int input_nodes, int output_nodes) {
    neural_net *net = (neural_net *)malloc(sizeof(neural_net));
    if (net == NULL) {
        fprintf(stderr, "Memory allocation failed.\n");
        return NULL;
    }

    net->input_nodes = input_nodes;
    net->output_nodes = output_nodes;
    net->weights = (float *)malloc(input_nodes * output_nodes * sizeof(float));
    net->biases = (float *)malloc(output_nodes * sizeof(float));
    if (net->weights == NULL || net->biases == NULL) {
        fprintf(stderr, "Memort allocation for weights and biases failed\n");
        free(net->weights);
        free(net->biases);
        free(net);
        return NULL;
    }

    srand(time(NULL));
    for (int i = 0; i < input_nodes * output_nodes; i++) {
        net->weights[i] = ((float)rand() / (float)(RAND_MAX)) * 0.01;
    }

    for (int i = 0; i < output_nodes; i++) {
        net->biases[i] = 0.0f;
    }

    return net;
}

void forward_propagation(neural_net *net, float *input, float *output) {
    for (int i = 0; i < net->output_nodes; i++) {
        output[i] = net->biases[i];
    }

    for (int out = 0; out < net->output_nodes; out++) {
        for (int in = 0; in < net->input_nodes; in++) {
            output[out] += input[in] * net->weights[out * net->input_nodes + in];
        }
    }
}

void softmax(float *output, int size) {
    float max = output[0];
    for (int i = 1; i < size; i++) {
        if (output[i] > max) {
            max = output[i];
        }
    }

    float sum = 0.0;
    for (int i = 0; i < size; i++) {
        output[i] = exp(output[i] - max);
        sum += output[i];
    }

    for (int i = 0; i < size; i++) {
        output[i] /= sum;
    }
}

void forward_propagate_with_activation(neural_net *net, float *input, float *output) {
    forward_propagation(net, input, output);
    softmax(output, net->output_nodes);
}

void free_network(neural_net *net) {
    if (net != NULL) {
        free(net->weights);
        free(net->biases);
        free(net);
    }
}
