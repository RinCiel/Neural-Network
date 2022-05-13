#pragma once
#include <iostream>
#include <vector>
#include <fstream>
#include <random>

struct image {
    int pixels[784]; // images are 28 x 28 pixels
};

class MNIST_Reader {
public:
    std::vector<int> training_labels;
    std::vector<unsigned char> training_labels_buffer;

    std::vector<image> training_images;
    std::vector<unsigned char> training_images_buffer;

    std::vector<int> test_labels;
    std::vector<unsigned char> test_labels_buffer;

    std::vector<image> test_images;
    std::vector<unsigned char> test_images_buffer;

    MNIST_Reader() {
        // check if files exist
        std::ifstream training_labels_file("train-labels-idx1-ubyte", std::ios::binary);
        std::ifstream training_images_file("train-images-idx3-ubyte", std::ios::binary);

        std::ifstream test_labels_file("t10k-labels-idx1-ubyte", std::ios::binary);
        std::ifstream test_images_file("t10k-images-idx3-ubyte", std::ios::binary);

        // check if files exist
        if (!training_labels_file.good() || !training_images_file.good() || !test_labels_file.good() || !test_images_file.good()) {
            std::cout << "MNIST files not found" << std::endl;
            return;
        }

        // read into buffer
        training_labels_buffer = std::vector<unsigned char>(std::istreambuf_iterator<char>(training_labels_file), {});
        training_images_buffer = std::vector<unsigned char>(std::istreambuf_iterator<char>(training_images_file), {});

        test_labels_buffer = std::vector<unsigned char>(std::istreambuf_iterator<char>(test_labels_file), {});
        test_images_buffer = std::vector<unsigned char>(std::istreambuf_iterator<char>(test_images_file), {});

        // close files
        training_labels_file.close();
        training_images_file.close();

        test_labels_file.close();
        test_images_file.close();
    }

    uint32_t readMagic(std::vector<unsigned char> &buffer) {
        uint32_t magic = 0;
        for (int i = 0; i < 4; i++) {
            magic = magic << 8;
            magic += buffer[i];
        }
        return magic;
    }

    void readTrainingData() {
        readTrainingLabels();
        readTrainingImages();
    }

    void readTrainingLabels() {
        uint32_t numberOfLabels = 0;
        for (int i = 4; i < 8; i++) {
            numberOfLabels = numberOfLabels << 8;
            numberOfLabels += training_labels_buffer[i];
        }

        // read labels
        for (int i = 8; i < 8 + numberOfLabels; i++) {
            training_labels.push_back(static_cast<int>(training_labels_buffer[i]));
        }
    }

    void readTrainingImages() {
        // get number of images
        uint32_t numberOfImages = 0;
        for (int i = 4; i < 8; i++) {
            numberOfImages = numberOfImages << 8;
            numberOfImages += training_images_buffer[i];
        }

        uint32_t numberOfRows = 0;
        for (int i = 8; i < 12; i++) {
            numberOfRows = numberOfRows << 8;
            numberOfRows += training_images_buffer[i];
        }

        uint32_t numberOfColumns = 0;
        for (int i = 12; i < 16; i++) {
            numberOfColumns = numberOfColumns << 8;
            numberOfColumns += training_images_buffer[i];
        }
        // read images
        int index = 0;
        for (int i = 16; i < 16 + numberOfImages * 784; i++) {
            struct image current;
            current.pixels[index] = static_cast<int>(training_images_buffer[i]);
            index++;
            if (index == 784) {
                index = 0;
                training_images.push_back(current);
            }
        }
    }

    void readTestData() {
        readTestLabels();
        readTestImages();
    }

    void readTestLabels() {
        // get number of labels
        uint32_t numberOfLabels = 0;
        for (int i = 4; i < 8; i++) {
            numberOfLabels = numberOfLabels << 8;
            numberOfLabels += test_labels_buffer[i];
        }

        // read labels
        for (int i = 8; i < 8 + numberOfLabels; i++) {
            test_labels.push_back(static_cast<int>(test_labels_buffer[i]));
        }
    }

    void readTestImages() {
        // get number of images
        uint32_t numberOfImages = 0;
        for (int i = 4; i < 8; i++) {
            numberOfImages = numberOfImages << 8;
            numberOfImages += test_images_buffer[i];
        }

        uint32_t numberOfRows = 0;
        for (int i = 8; i < 12; i++) {
            numberOfRows = numberOfRows << 8;
            numberOfRows += test_images_buffer[i];
        }

        uint32_t numberOfColumns = 0;
        for (int i = 12; i < 16; i++) {
            numberOfColumns = numberOfColumns << 8;
            numberOfColumns += test_images_buffer[i];
        }

        // read images
        int index = 0;
        for (int i = 16; i < 16 + numberOfImages * 784; i++) {
            struct image current;
            current.pixels[index] = static_cast<int>(test_images_buffer[i]);
            index++;
            if (index == 784) {
                index = 0;
                test_images.push_back(current);
            }
        }
    }

    std::vector<int> randomLabels;
    std::vector<image> randomImages;

    void randomTrainingData(int amount) {
        for (int i = 0; i < amount; i++) {
            int index = rand() % training_labels.size();
            randomLabels.push_back(training_labels[index]);
            randomImages.push_back(training_images[index]);
        }
    }

    void randomTestData(int amount) {
        for (int i = 0; i < amount; i++) {
            int index = rand() % test_labels.size();
            randomLabels.push_back(test_labels[index]);
            randomImages.push_back(test_images[index]);
        }
    }
};
