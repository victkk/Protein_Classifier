import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import label_binarize
from fea import feature_extraction
from sklearn.svm import SVC
from Bio.PDB import PDBParser

from MyNptorch.mynptorch.core.tensor import Tensor
from utils import create_timestamped_folder
import logging
from tqdm import tqdm


class SVMModel:
    """
    Initialize Support Vector Machine (SVM from sklearn) model.

    Parameters:
    - C (float): Regularization parameter. Default is 1.0.
    - kernel (str): Specifies the kernel type to be used in the algorithm. Default is 'rbf'.
    """

    def __init__(self, C=1.0, kernel="rbf"):
        self.model = SVC(C=C, kernel=kernel, probability=True)

    def train(self, train_data, train_targets):
        """
        Train the Support Vector Machine model.

        Parameters:
        - train_data (array-like): Training data.
        - train_targets (array-like): Target values for the training data.
        """
        self.model.fit(train_data, train_targets)

    def evaluate(self, data, targets):
        """
        Evaluate the performance of the Support Vector Machine model.

        Parameters:
        - data (array-like): Data to be evaluated.
        - targets (array-like): True target values corresponding to the data.

        Returns:
        - float: Accuracy score of the model on the given data.
        """
        return self.model.score(data, targets)


class SVMFromScratch:
    def __init__(self, lr=0.001, num_iter=200, c=0.1, save_dir=None, log_interval=100):
        self.lr = lr
        self.num_iter = num_iter
        self.weights = None
        self.bias = 0
        self.C = c
        self.mean = None
        self.std = None
        self.save_dir = save_dir
        self.log_interval = log_interval
        if self.save_dir:
            print(f"logging to{self.save_dir}")
            logging.basicConfig(
                level=logging.INFO,
                format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                filename=os.path.join(self.save_dir, "training.log"),
                filemode="w",
            )

    def compute_loss(self, y, predictions):
        """
        SVM Loss function:
        hinge_loss = 1/2 * ||w||^2 + C * sum(max(0, 1 - y * z))
        """
        num_samples = y.data.shape[0]
        # 1. Hinge Loss: max(0, 1 - y * z)
        hinge_loss = Tensor.clamp(1 - y * predictions, min=0)
        # hinge_loss_sum = Tensor.sum(
        #     hinge_loss * (1 - y) / 2 * self.y1_percentage
        #     + (1 - self.y1_percentage) / 2 * hinge_loss * (1 + y) / 2
        # )  # sum
        hinge_loss_sum = Tensor.sum(hinge_loss)
        # 2. Regularization term: 1/2 * ||w||^2
        regularization = 0.5 * Tensor.sum(self.weights**2)
        total_loss = self.C * hinge_loss_sum + regularization
        loss = total_loss / num_samples
        return loss

    def standardize(self, X):
        return (X - self.mean) / self.std

    # def initialize_weights(self, n_features):
    #     self.weights = Tensor(np.random.rand(1, n_features) / 100, requires_grad=True)
    #     self.bias = Tensor(np.random.rand(1) / 10, requires_grad=True)

    def initialize_weights(self, n_features):
        self.weights = Tensor(np.zeros((1, n_features)), requires_grad=True)
        self.bias = Tensor(np.zeros(1), requires_grad=True)

    #  todo:
    def train(
        self,
        train_data,
        train_targets,
        visualize_backward_graph=False,
        lr=None,
        num_iter=None,
    ):
        self.num_iter = num_iter if num_iter else self.num_iter
        X = np.array(train_data)
        num_features, num_samples = X.shape
        y = np.array(train_targets)
        self.y1_percentage = max(0.1, np.sum(y == 1) / y.shape[0])
        # y = np.where(y == 0, -1, 1)
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
        self.std[self.std == 0] = 1e-3
        X = self.standardize(X)

        X = Tensor(X.T, requires_grad=False)
        y = Tensor(y, requires_grad=False)

        pbar = tqdm(range(self.num_iter), desc="Training", unit="epoch")
        for iteration in pbar:

            predictions = self.weights @ X + self.bias
            loss = self.compute_loss(y, predictions)
            if visualize_backward_graph:
                if self.save_dir and not os.path.exists(
                    os.path.join(self.save_dir, "compute_graph.png")
                ):
                    loss.visualize_backward(
                        filename=os.path.join(self.save_dir, "compute_graph")
                    )
            loss.backward()
            self.weights -= self.lr * self.weights.grad
            self.bias -= self.lr * np.mean(self.bias.grad)
            self.weights.grad_fn = None
            self.bias.grad_fn = None

            if (
                self.save_dir
                and self.log_interval != -1
                and iteration % self.log_interval == 0
            ):
                logging.info(f"\tLoss: {loss.data:.4f}")

            pbar.set_postfix_str(
                f"Loss: \033[1;36m{loss.data:.4f}\033[0m | "
                f"ETA: {pbar.format_interval(pbar.format_dict['elapsed']/(iteration+1)*(self.num_iter-iteration))}"
            )

    def predict(self, X):
        # sign
        X = self.standardize(X)
        svm_model = self.weights @ X.T + self.bias
        predictions = np.sign(svm_model.data)
        return predictions

    def evaluate(
        self, data, targets, show_confusion_matrix=True, dataset_id=0, istest=False
    ):
        X = np.array(data)
        y = np.array(targets)

        # y = np.where(y == 0, -1, 1)
        predictions = self.predict(X)
        predictions = np.sign(predictions.data)
        tp = np.sum((predictions == 1) & (y == 1))
        tn = np.sum((predictions == -1) & (y == -1))
        fp = np.sum((predictions == 1) & (y == -1))
        fn = np.sum((predictions == -1) & (y == 1))
        if show_confusion_matrix:
            # Create confusion matrix visualization
            self.draw_confusion_matrix(tp, tn, fp, fn, dataset_id, istest)
            if not os.path.exists(
                os.path.join(self.save_dir, "confusion_matrix_dataset")
            ):
                os.makedirs(os.path.join(self.save_dir, "confusion_matrix_dataset"))
            plt.savefig(
                os.path.join(
                    self.save_dir,
                    "confusion_matrix_dataset",
                    f"dataset{dataset_id}_{'test' if istest else 'train'}.png",
                )
            )
            plt.close()
        return np.mean(predictions == y)

    def draw_confusion_matrix(self, tp, tn, fp, fn, dataset_id, istest):
        confusion_matrix = np.array([[tn, fp], [fn, tp]])

        plt.figure(figsize=(6, 5))
        plt.imshow(confusion_matrix, interpolation="nearest", cmap=plt.cm.Blues)
        plt.title(
            "Confusion Matrix"
            + f" for dataset {dataset_id}"
            + f"{'(Test)' if istest else '(Train)'}"
        )
        plt.colorbar()
        classes = ["0", "1"]
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, classes)
        plt.yticks(tick_marks, classes)
        plt.xlabel("Predicted Label")
        plt.ylabel("True Label")

        # Add text annotations
        thresh = confusion_matrix.max() / 2.0
        for i in range(2):
            for j in range(2):
                plt.text(
                    j,
                    i,
                    format(confusion_matrix[i, j], "d"),
                    ha="center",
                    va="center",
                    color="white" if confusion_matrix[i, j] > thresh else "black",
                )


def data_preprocess(args):
    if args.ent:
        diagrams = feature_extraction()[0]
    else:
        diagrams = np.load("./data/diagrams.npy")
    cast = pd.read_table("./data/SCOP40mini_sequence_minidatabase_19.cast")
    cast.columns.values[0] = "protein"
    # standardize the data
    mean = np.mean(diagrams, axis=0)
    std = np.std(diagrams, axis=0)
    diagrams = (diagrams - mean) / (std + 1e-8)

    data_list = []
    target_list = []
    for task in range(1, 56):  # Assuming only one task for now
        task_col = cast.iloc[:, task].to_numpy()
        train_mask = (task_col == 1) | (task_col == 2)
        test_mask = (task_col == 3) | (task_col == 4)
        train_data = diagrams[train_mask]
        test_data = diagrams[test_mask]
        train_targets = task_col[train_mask] * (-2) + 3
        test_targets = task_col[test_mask] * (-2) + 7
        assert train_targets.shape[0] == train_data.shape[0]
        assert test_targets.shape[0] == test_data.shape[0]
        data_list.append((train_data, test_data))
        target_list.append((train_targets, test_targets))

    return data_list, target_list


def main(args):

    data_list, target_list = data_preprocess(args)

    task_acc_train = []
    task_acc_test = []

    ## You can also consider other different settings
    # model = SVMModel(C=args.C,kernel=args.kernel)
    save_dir = create_timestamped_folder(args.description, base_dir="./results")

    if not args.custom:
        model = SVMModel(C=args.C, kernel=args.kernel)
    else:
        model = SVMFromScratch(
            save_dir=save_dir, lr=0.01, log_interval=100, num_iter=1000
        )

    for i in range(len(data_list)):
        train_data, test_data = data_list[i]
        train_targets, test_targets = target_list[i]

        print(f"Processing dataset {i+1}/{len(data_list)}")
        if args.custom:
            model.initialize_weights(train_data.shape[1])
        # Train the model
        if not args.custom:
            model.train(train_data, train_targets)
        else:
            model.train(train_data, train_targets, visualize_backward_graph=True)

        # Evaluate the model
        if not args.custom:
            train_accuracy = model.evaluate(train_data, train_targets)
            test_accuracy = model.evaluate(test_data, test_targets)
        else:
            train_accuracy = model.evaluate(train_data, train_targets, dataset_id=i)
            test_accuracy = model.evaluate(
                test_data, test_targets, dataset_id=i, istest=True
            )

        print(
            f"Dataset {i+1}/{len(data_list)} - Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}"
        )

        task_acc_train.append(train_accuracy)
        task_acc_test.append(test_accuracy)

    print("Training accuracy:", sum(task_acc_train) / len(task_acc_train))
    print("Testing accuracy:", sum(task_acc_test) / len(task_acc_test))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="SVM Training and Evaluation")
    parser.add_argument("--C", type=float, default=1.0, help="Regularization parameter")
    parser.add_argument(
        "--ent",
        action="store_true",
        help="Load data from a file using a feature engineering function feature_extraction() from fea.py",
    )
    parser.add_argument(
        "--custom",
        action="store_true",
        help="use self implemented logistic regression model if present",
    )
    parser.add_argument(
        "--description",
        help="description of the experiment, will be included in the result folder name",
    )
    parser.add_argument("--kernel", type=str, default="rbf", help="Kernel type for SVM")
    args = parser.parse_args()
    main(args)
