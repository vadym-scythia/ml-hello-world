#include "stdio.h"
#include "main.h"
#include "stdlib.h"
#include "time.h"
#include "math.h"

int main() {
    const char *train_image_filename = "dataset/train-images.idx3-ubyte";
    const char *train_label_filename = "dataset/train-labels.idx1-ubyte";
    const char *test_image_filename = "dataset/t10k-images.idx3-ubyte";
    const char *test_label_filename = "dataset/t10k-labels.idx1-ubyte";

    mnist_data *train_images = load_mnist_images(train_image_filename);
    mnist_data *train_labels = load_mnist_labels(train_label_filename);
    mnist_data *test_images = load_mnist_images(test_image_filename);
    mnist_data *test_labels = load_mnist_labels(test_label_filename);

    if (train_images && train_labels && test_images && test_labels) {
        printf("Images and labels loaded successfully.\n");
        printf("Number of training images: %d\n", train_images->size);
        printf("Number of training labels: %d\n", train_labels->size);
        printf("Number of testing images: %d\n", test_images->size);
        printf("Number of testing labels: %d\n", test_labels->size);

        neural_net *net = initialize_network(784, 10);

        train(net, train_images, train_labels, 30, 0.1); // For testing, use 1 epoch and a learning rate of 0.1

        float accuracy = evaluate(net, test_images, test_labels);
        printf("Accuracy on test set: %.2f%%\n", accuracy * 100);

        free_network(net);

        for (int i = 0; i < train_images->size; i++) {
            free(train_images->images[i]);
        }
        free(train_images->images);
        free(train_images);

        free(train_labels->labels);
        free(train_labels);

        for (int i = 0; i < test_images->size; i++) {
            free(test_images->images[i]);
        }
        free(test_images->images);
        free(test_images);

        free(test_labels->labels);
        free(test_labels);
    } else {
        printf("Failed to load images and labels.\n");
    }

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

float cross_entropy_loss(float *output, int *target, int size) {
    float loss = 0.0f;
    for (int i = 0; i < size; i++) {
        if (target[i] == 1) {
            loss -= log(output[i]);
        }
    }

    return loss;
}

void backpropagation(neural_net *net, float *input, int *target, float *output, float learning_rate) {
    float *delta = (float *)malloc(net->output_nodes * sizeof(float));

    for (int i = 0; i < net->output_nodes; i++) {
        delta[i] = output[i] - target[i];
    }

    for (int i = 0; i < net->output_nodes; i++) {
        for (int j = 0; j < net->input_nodes; j++) {
            net->weights[i * net->input_nodes + j] -= learning_rate * delta[i] * input[j];
        }
        net->biases[i] -= learning_rate * delta[i];
    }

    free(delta);
}

void train(neural_net *net, mnist_data *train_data, mnist_data *train_labels, int epochs, float learning_rate) {
    for (int epoch = 0; epoch < epochs; epoch++) {
        float total_loss = 0.0;
        for (int i = 0; i < train_data->size; i++) {
            float *ouput = (float *)malloc(net->output_nodes * sizeof(float));
            
            forward_propagate_with_activation(net, train_data->images[i], ouput);
            
            int target[10] = {0};
            target[train_labels->labels[i]] = 1;

            total_loss += cross_entropy_loss(ouput, target, net->output_nodes);

            backpropagation(net, train_data->images[i], target, ouput, learning_rate);

            free(ouput);
        }
        printf("Epoch %d, Loss: %f\n", epoch + 1, total_loss / train_data->size);
    }
}

float evaluate(neural_net *net, mnist_data *test_data, mnist_data *test_labels) {
    int correct_predictions = 0;

    for (int i = 0; i < test_data->size; i++) {
        float *output = (float *)malloc(net->output_nodes * sizeof(float));

        forward_propagate_with_activation(net, test_data->images[i], output);

        int predicted_label = 0;
        for (int j = 1; j < net->output_nodes; j++) {
            if (output[j] > output[predicted_label]) {
                predicted_label = j;
            }
        }

        if (predicted_label == test_data->labels[i]) {
            correct_predictions++;
        }

        free(output);
    }

    return (float)correct_predictions / test_data->size;
}

void free_network(neural_net *net) {
    if (net != NULL) {
        free(net->weights);
        free(net->biases);
        free(net);
    }
}
