import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST
from decision_tree import DecisionTree
import time
import os
from datetime import datetime

def plot_image(image_list):
    image_array = np.array(image_list).reshape(28, 28)
    plt.figure()
    plt.imshow(image_array, cmap='gray_r')
    plt.show()

def load_weights(file_name, num_lines, dim):
    w = np.empty((num_lines, dim))
    b = np.empty((num_lines, 1))

    with open(file_name, 'r') as f:
        for idx, line in enumerate(f):
            values = np.fromstring(line.strip(), sep=',')
            print(f"Shape of values at index {idx}: {values.shape}")
            w[idx, :] = values[:dim]
            b[idx, 0] = values[dim]

    return w, b

def my_load_weights(file_name):
    return np.loadtxt(file_name, delimiter=',')

def treat_images(all_images, all_labels, weight, lim=np.inf):
    normalized_images = all_images.astype(np.float32) / 255.0
    labels_array = np.array(all_labels, dtype=np.float32)

    bias_column = np.ones((normalized_images.shape[0], 1))
    processed_images = np.hstack((bias_column, normalized_images))

    if len(processed_images) > lim:
        processed_images = processed_images[:lim, :]
        labels_array = labels_array[:lim]

    transformed_images = processed_images @ weight.T
    return transformed_images, labels_array

if __name__ == "__main__":
    script_start_time = time.time()

    # Locate the dataset directory
    abs_path = os.path.dirname(__file__)
    dataset_dir = "datasets/MNIST/raw"
    dataset_path = os.path.join(abs_path, dataset_dir)

    # Load MNIST dataset
    print("Loading MNIST data...")
    mnist_loader = MNIST(dataset_path)
    training_images, training_labels = mnist_loader.load_training()
    testing_images, testing_labels = mnist_loader.load_testing()

    # Loading weights
    abs_weights_path = os.path.dirname(__file__)
    weights_file = "weights_Lizarralde.txt"
    weights_path = os.path.join(abs_weights_path, weights_file)

    weights = my_load_weights(weights_path)

    # Prepare training and test data
    processed_train_images, processed_train_labels = treat_images(training_images, training_labels, weights)
    processed_test_images, processed_test_labels = treat_images(testing_images, testing_labels, weights)

    print(f"Training images shape: {processed_train_images.shape}")
    print(f"Training labels shape: {processed_train_labels.shape}")
    print(f"Test images shape: {processed_test_images.shape}")
    print(f"Test labels shape: {processed_test_labels.shape}")

    # Hyperparameters
    leaf_limits = np.array([400])
    threshold_checks = [100]

    # Timestamp for file creation
    current_time = datetime.now().strftime("%Y-%m-%d-%H-%M-%S")
    log_filename = f"tests/test-{current_time}.txt"

    with open(log_filename, 'ab') as log_file:
        for max_leaves in leaf_limits:
            for max_thresholds in threshold_checks:
                iteration_start = time.time()

                print(f"Max leaves: {max_leaves}")
                print(f"Max thresholds: {max_thresholds}")

                # Train and test the decision tree
                dt = DecisionTree()
                dt.train(processed_train_images, processed_train_labels, max_leaves, max_thresholds)
                dt.test(processed_test_images, processed_test_labels)

                # Capture accuracy
                training_accuracy = dt.get_train_accuracy()
                testing_accuracy = dt.get_test_accuracy()

                # Measure time for this iteration
                iteration_duration = time.time() - iteration_start
                result_data = np.array([[max_leaves, max_thresholds, iteration_duration, training_accuracy, testing_accuracy]])

                # Save the results
                np.savetxt(log_file, result_data, fmt='%1.10g', delimiter=',')

    # Total execution time
    total_duration = time.time() - script_start_time
    minutes, seconds = divmod(total_duration, 60)
    print(f"Total time to run script: {int(minutes)} min {round(seconds, 1)} sec")

    # plot_accuracies  # You can uncomment this if plotting is needed
