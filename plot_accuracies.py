import matplotlib.pyplot as plt
import numpy as np
import os
import glob
import sys

def plot_acc(leaves, acc_train, acc_test):
    fig, ax = plt.subplots()
    
    # Plotting train and test accuracy with custom markers and labels
    ax.plot(leaves, acc_train, marker='o', linestyle='-', color='r', label='Train Accuracy')
    ax.plot(leaves, acc_test, marker='o', linestyle='-', color='g', label='Test Accuracy')
    
    # Adding a horizontal line for accuracy threshold
    ax.axhline(y=90, color='b', linestyle='-', label='Accuracy Threshold (90%)')
    
    # Setting titles and labels
    ax.set_title("Training vs Testing Accuracy")
    ax.set_xlabel("Number of Leaves")
    ax.set_ylabel("Accuracy (%)")
    ax.set_xlim([0, 1100])

    # Displaying the grid and legend
    ax.grid(True)
    ax.legend()

    plt.show()


def plot_thresh(thresh, acc_train, acc_test, time):
    fig, ax1 = plt.subplots()

    # Plot for accuracy on the primary y-axis
    ax1.plot(thresh, acc_train, marker='o', linestyle='-', color='r', label='Train Accuracy')
    ax1.plot(thresh, acc_test, marker='o', linestyle='-', color='g', label='Test Accuracy')
    ax1.set_xlabel('Number of Thresholds')
    ax1.set_ylabel('Accuracy (%)')

    # Plot for time on the secondary y-axis
    ax2 = ax1.twinx()
    ax2.plot(thresh, time, marker='o', linestyle='-', color='b', label='Time Consumption')
    ax2.set_ylabel('Time (s)', color='b')
    ax2.tick_params(axis='y', labelcolor='b')

    # Adding a title and grid
    fig.suptitle("Accuracy vs Time Based on Thresholds")
    ax1.grid(True)

    # Combining legends from both axes
    fig.legend(loc='upper left', bbox_to_anchor=(0.1, 0.85))

    plt.show()


def plot_data(data):
    # Extract columns from data
    leaves = data[:, 0]
    thresholds = data[:, 1]
    time = data[:, 2]
    acc_train = data[:, 3]
    acc_test = data[:, 4]

    # Create sets to determine whether to plot against leaves or thresholds
    set_of_thresh = np.unique(thresholds)
    set_of_leaves = np.unique(leaves)

    if len(set_of_leaves) > len(set_of_thresh):
        xdata = leaves
        plot_label = "Leaves"
    else:
        xdata = thresholds
        plot_label = "Thresholds"

    plt.figure()

    # Plot accuracy against the chosen variable
    plt.plot(xdata, acc_train, marker='o', linestyle='-', label='Train Accuracy')
    plt.plot(xdata, acc_test, marker='o', linestyle='-', label='Test Accuracy')
    plt.axhline(y=90, color='b', linestyle='-', label='Accuracy Threshold')

    plt.title(f"Train vs Test Accuracy by {plot_label}")
    plt.xlabel(plot_label)
    plt.ylabel("Accuracy (%)")
    plt.grid(True)
    plt.legend()

    plt.show()


if __name__ == "__main__":
    # Determine the absolute path to locate the test files
    base_dir = os.path.dirname(__file__)
    test_files = glob.glob(os.path.join(base_dir, "tests/*.txt"))

    if not test_files:
        print("No test files found, exiting...")
        sys.exit()

    # Sort files by creation time, most recent first
    test_files.sort(key=os.path.getctime, reverse=True)

    # If an argument is passed, use it as an offset to select a different file
    file_offset = int(sys.argv[1]) if len(sys.argv) > 1 else 0

    # Select the appropriate file and print its path
    selected_file = test_files[file_offset]
    print(f"Selected test file: {selected_file}")

    # Load the selected file's data
    data = np.loadtxt(selected_file, delimiter=",", skiprows=0)

    # Extract relevant data for plotting
    num_leaves = data[:, 0]
    thresholds = data[:, 1]
    time_spent = data[:, 2]
    acc_train = data[:, 3]
    acc_test = data[:, 4]

    print(f"Threshold values: {thresholds}")

    # Plot accuracy vs number of leaves
    plot_acc(num_leaves, acc_train, acc_test)

    # Uncomment if you want to also plot the threshold-based accuracy and time
    # plot_thresh(thresholds, acc_train, acc_test, time_spent)
