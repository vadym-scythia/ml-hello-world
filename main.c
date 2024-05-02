#include "stdio.h"
#include "main.h"
#include "stdlib.h"

int main() {
    const char *images_filename = "dataset\\train-images.idx3-ubyte";
    const char *labels_filename = "dataset\\train-labels.idx1-ubyte";

    mnist_data *train_images = load_mnist_images(images_filename);
    mnist_data *train_labels = load_mnist_labels(labels_filename);
    if (train_images && train_labels) {
        printf("Images and labels loaded successfully.\n");
        printf("Number of images: %d\n", train_images->size);
        printf("Number of labels: %d\n", train_labels->size);
        printf("First image label: %d\n", train_labels->labels[0]);
        for (int i = 0; i < 28; i++) {
            for (int j = 0; j < 28; j++) {
                if (train_images->images[0][i * 28 + j] == 0.00)
                    printf("%c ", "O"); // %c used to make digits more visible try
                else
                    printf("%c ", "X");
            }
            printf("\n");
        }
    } else {
        printf("Failed to load images and labels. \n");
    }

    if (train_images) {
        for (int i = 0; i < train_images->size; ++i) {
            free(train_images->images[i]);
        }
        free(train_images->images);
        free(train_images);
    }

    if (train_labels) {
        free(train_labels->labels);
        free(train_labels);
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
