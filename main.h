/*
MNIST section.
MNIST file formats:
Images: The first 16 bytes are the header, the next bytes are pixel values.
    Offset 0: magic number (should be 2051)
    Offset 4: number of images
    Offset 8: number of rows
    Offset 12: number of columns
    Offset 16: image pixel data begins
Labels: The first 8 bytes are the header, the next bytes are the labels.
    Offset 0: magic number (should be 2049)
    Offset 4: number of items
    Offset 8: label data begins
*/
typedef struct {
    int size;
    int *labels;
    float **images;
} mnist_data;

int reverse_int(int i);
mnist_data* load_mnist_images(const char* images_filename);
mnist_data* load_mnist_labels(const char* labels_filename);

// Neural Network architecture section.
typedef struct {
    int input_nodes;
    int output_nodes;
    float *weights;
    float *biases;
} neural_net;

neural_net* initialize_network(int input_nodes, int output_nodes);

// Propagation section.
void forward_propagation(neural_net *net, float *input, float *output);
void softmax(float *input, int size);
void forward_propagate_with_activation(neural_net *net, float *input, float *output);

void backpropagation(neural_net *net, float *input, int *target, float *output, float learning_rate);
void train(neural_net *net, mnist_data *train_data, mnist_data *train_labels, int epochs, float learning_rate);
float cross_entropy_loss(float *output, int *target, int size);

// Action section.
float evaluate(neural_net *net, mnist_data *test_data, mnist_data *test_labels);
//TODO save wieghts? function

// Matrix arithmetics section. Optional as separate functions.
void matrix_multiply(float *A, float *B, float *C, int A_rows, int A_cols, int B_cols);
void matrix_add(float *A, float *B, int size);

//Free functions
void free_network(neural_net *net);
