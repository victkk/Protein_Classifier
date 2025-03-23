import os
import argparse
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import label_binarize
from fea import feature_extraction
from sklearn.linear_model import LogisticRegression
from Bio.PDB import PDBParser
import matplotlib.pyplot as plt
from MyNptorch.mynptorch.core.tensor import Tensor
import time
from tqdm import tqdm
from utils import create_timestamped_folder
import logging


class LRModel:
    def __init__(self):
        self.model = LogisticRegression(max_iter=10000, penalty=None)

    def train(self, train_data, train_targets):
        self.model.fit(train_data, train_targets)

    def evaluate(self, data, targets):
        preds = self.model.predict(data)
        return accuracy_score(targets, preds)


class LRFromScratch:

    def __init__(
        self, lr=1e-1, l1_lambda=0, l2_lambda=0, save_dir=None, log_interval=100
    ):
        """
        l1_lambda: L1 regularization lambda
        l2_lambda: L2 regularization lambda
        save_dir: directory to save the log
        log_interval: -1 for no logging
        """
        self.l1_lambda = l1_lambda
        self.l2_lambda = l2_lambda
        self.weights = Tensor(np.random.rand(1, 300) / 100, requires_grad=True)
        self.bias = Tensor(np.random.rand(1) / 10, requires_grad=True)
        self.lr = lr
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

    def _loss(self, y_pred, y):
        epsilon = 1e-10
        loss = y * Tensor.log(y_pred + epsilon) + (1 - y) * Tensor.log(
            1 - y_pred + epsilon
        )
        return -Tensor.mean(loss)

    def _l1_loss(self):
        # bug: auto backward pass not working!
        if self.l1_lambda != 0:
            raise NotImplementedError
            return self.l1_lambda * Tensor.sum(self.weights**2**0.5)
        return 0

    def _l2_loss(self):

        # bug: auto backward pass not working!
        if self.l2_lambda != 0:
            raise NotImplementedError
            return self.l2_lambda * Tensor.sum(self.weights**2)
        return 0

    def initialize_weights(self, n_features):
        self.weights = Tensor(np.random.rand(1, n_features) / 100, requires_grad=True)
        self.bias = Tensor(np.random.rand(1) / 10, requires_grad=True)

    def backward(self, weight, bias, x, y):
        # only for debugging auto backward pass
        # not used in training
        N = x.shape[1]
        z = weight @ x + bias
        y_pred5 = 1 / (1 + np.exp(-z))
        epsillon = 1e-10
        dl_dz = -1 / N * ((1 - y_pred5 + epsillon) * y - (y_pred5 + epsillon) * (1 - y))
        dl_dw = dl_dz @ x.T
        dl_db = np.sum(dl_dz)
        return dl_dw, dl_db

    def train(
        self,
        train_data,
        train_targets,
        epochs=10000,
        visualize_backward_graph=True,
        lr=None,
    ):
        if lr:
            self.lr = lr
        train_data = Tensor(train_data.T)
        train_targets = Tensor(train_targets)
        pbar = tqdm(range(epochs), desc="Training", unit="epoch")
        for eps in pbar:
            y_predict1 = self.weights @ train_data + self.bias
            y_predict2 = 1 / (1 + Tensor.exp(-y_predict1))
            loss = (
                self._loss(y_predict2, train_targets)
                + self._l1_loss()
                + self._l2_loss()
            )
            loss.backward()
            self.weights -= self.lr * self.weights.grad
            self.bias -= np.mean(self.lr * self.bias.grad)
            # TODO: implement zero_grad
            # hack: don't remove this line
            self.bias.grad_fn = None
            self.weights.grad_fn = None

            # for logging
            pbar.set_postfix_str(
                f"Loss: \033[1;36m{loss.data:.4f}\033[0m | "
                f"ETA: {pbar.format_interval(pbar.format_dict['elapsed']/(eps+1)*(epochs-eps))}"
            )
            if visualize_backward_graph:
                if self.save_dir:
                    loss.visualize_backward(
                        filename=os.path.join(self.save_dir, "compute_graph")
                    )
                else:
                    loss.visualize_backward()
            if (
                self.save_dir
                and self.log_interval != -1
                and eps % self.log_interval == 0
            ):
                logging.info(f"\tLoss: {loss.data:.4f}")

    def evaluate(
        self, data, targets, istest=True, dataset_id=0, show_confusion_matrix=False
    ):
        """
        Evaluate the performance of the Logistic Regression model.

        Parameters:
        - data (array-like): Data to be evaluated.
        - targets (array-like): True target values corresponding to the data.

        Returns:
        - float: Accuracy score of the model on the given data.
        """
        # Convert data and targets to Tensor
        data = Tensor(data.T)
        targets = Tensor(targets)

        # Make predictions
        y_pred = self.weights @ data + self.bias
        y_pred = 1 / (1 + Tensor.exp(-y_pred))  # Apply sigmoid to get probabilities
        # Convert probabilities to binary predictions (0 or 1)
        y_pred_binary = (y_pred.data >= 0.5).astype(int)
        tp = ((y_pred_binary == 1) & (targets.data == 1)).sum()
        tn = ((y_pred_binary == 0) & (targets.data == 0)).sum()
        fp = ((y_pred_binary == 1) & (targets.data == 0)).sum()
        fn = ((y_pred_binary == 0) & (targets.data == 1)).sum()
        if show_confusion_matrix:
            # Create confusion matrix visualization
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
            # plt.show()
            # save plt to file
            plt.savefig(
                os.path.join(
                    self.save_dir,
                    "confusion_matrix_dataset",
                    f"dataset{dataset_id}_{'test' if istest else 'train'}.png",
                )
            )

        # Calculate accuracy
        correct = (y_pred_binary == targets.data).sum()
        total = targets.data.shape[0]
        accuracy = correct / total
        logging.info(
            f"\t{'test' if istest else 'train'} accuracy:{accuracy} tp:{tp} tn:{tn} fp:{fp} fn:{fn}"
        )
        return accuracy


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
        train_targets = task_col[train_mask] * (-1) + 2
        test_targets = task_col[test_mask] * (-1) + 4
        assert train_targets.shape[0] == train_data.shape[0]
        assert test_targets.shape[0] == test_data.shape[0]
        data_list.append((train_data, test_data))
        target_list.append((train_targets, test_targets))

    return data_list, target_list


def main(args):

    data_list, target_list = data_preprocess(args)

    task_acc_train = []
    task_acc_test = []

    # model = LRModel()
    save_dir = create_timestamped_folder(args.description, base_dir="./results")
    model = LRFromScratch(save_dir=save_dir, log_interval=100)
    for i in range(len(data_list)):
        train_data, test_data = data_list[i]
        train_targets, test_targets = target_list[i]

        logging.info(f"Processing dataset {i+1}/{len(data_list)}")
        model.initialize_weights(train_data.shape[1])
        model.train(
            train_data,
            train_targets,
            visualize_backward_graph=False,
            lr=5e-1,
        )

        # Evaluate the model
        train_accuracy = model.evaluate(
            train_data, train_targets, istest=False, dataset_id=i + 1
        )
        test_accuracy = model.evaluate(
            test_data,
            test_targets,
            istest=True,
            dataset_id=i + 1,
            show_confusion_matrix=True,
        )

        print(
            f"Dataset {i+1}/{len(data_list)} - Train Accuracy: {train_accuracy}, Test Accuracy: {test_accuracy}"
        )
        task_acc_train.append(train_accuracy)
        task_acc_test.append(test_accuracy)
    logging.info(f"Training accuracy: {sum(task_acc_train) / len(task_acc_train)}")
    logging.info(f"Testing accuracy: {sum(task_acc_test) / len(task_acc_test)}")
    print("Training accuracy:", sum(task_acc_train) / len(task_acc_train))
    print("Testing accuracy:", sum(task_acc_test) / len(task_acc_test))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="LR Training and Evaluation")
    parser.add_argument(
        "--ent",
        action="store_true",
        help="Load data from a file using a feature engineering function feature_extraction() from fea.py",
    )
    parser.add_argument(
        "--description",
        help="description of the experiment, will be included in the result folder name",
    )
    args = parser.parse_args()
    main(args)
