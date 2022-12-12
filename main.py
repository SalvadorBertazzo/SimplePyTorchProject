import torch
from torch import nn
import matplotlib.pyplot as plt
import warnings


# <*> VISUALIZE DATA
def plot_predictions(train_data,
                     train_labels,
                     test_data,
                     test_labels,
                     predictions=None):
    plt.figure(figsize=(10, 7))

    plt.scatter(train_data, train_labels, c='b', s=4, label='Training data')
    plt.scatter(test_data, test_labels, c='g', s=4, label='Testing data')

    if predictions is not None:
        plt.scatter(test_data, predictions, c='y', s=4, label='Predictions')

    plt.legend(prop={"size": 14})
    plt.show()


# <*> MODEL
class LinearRegressionModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.weight = nn.Parameter(torch.rand(1,
                                              requires_grad=True,
                                              dtype=torch.float))
        self.bias = nn.Parameter(torch.rand(1,
                                            requires_grad=True,
                                            dtype=torch.float))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.weight * x + self.bias


# <*> MAIN FUNCTION
def main():
    warnings.filterwarnings("ignore", category=DeprecationWarning)  # Hide deprecated warnings
    # <1> GENERATE DATASET (Basic DS for educational purposes)
    # 1. Results that we are looking for
    weight = 0.7
    bias = 0.3

    # 2. Create tensors
    start = 0
    end = 1
    step = 0.02

    X = torch.arange(start, end, step).unsqueeze(dim=1)
    y = weight * X + bias

    # 3. Split tensors to get train and test data (80/20)
    train_split = int(0.8 * len(X))
    X_train = X[:train_split]
    y_train = y[:train_split]
    X_test = X[train_split:]
    y_test = y[train_split:]

    # <2> CREATE MODEL
    # 1. "Disable" randomness
    torch.manual_seed(42)
    # 2. Create model
    model = LinearRegressionModel()
    # 3. Create loss function
    loss_fn = nn.L1Loss()
    # 4. Create optimizer
    optimizer = torch.optim.SGD(params=model.parameters(), lr=0.01)

    # <3> TRAIN MODEL
    # *. Get model info before training it
    y_pred_old = model(X_test)
    weight_old = round(model.state_dict()['weight'].item(), 4)
    bias_old = round(model.state_dict()['bias'].item(), 4)

    # 1. Set the number of epochs
    epochs = 100

    # 2. Train
    for epoch in range(epochs):
        model.train()
        # 1. Forward pass
        y_pred = model(X_train)
        # 2. Calculate loss
        loss = loss_fn(y_pred, y_train)
        # 3. Zero grad optimizer
        optimizer.zero_grad()  # Reset optimizer values in each iteration
        loss.backward()
        optimizer.step()

        model.eval()

    # <4> EVAL MODEL
    with torch.inference_mode():
        # 1. Get predictions with model already trained
        y_pred_new = model(X_test)

        # 2. Show info and graphics
        print("-" * 100)
        print(f"\tThe model learned the following values -> *Weight: {round(model.state_dict()['weight'].item(), 4)}"
              f"  *Bias: {round(model.state_dict()['bias'].item(), 4)}")
        plot_predictions(X_train, y_train, X_test, y_test)

        print(f"\tThis was the model predictions before training it -> *Weight: {weight_old}  *Bias: {bias_old}")
        plot_predictions(X_train, y_train, X_test, y_test, predictions=y_pred_old)

        print(f"\tThe original values are -> *Weight: {weight}  *Bias: {bias}")
        plot_predictions(X_train, y_train, X_test, y_test, predictions=y_pred_new)
        print("-" * 100)


if __name__ == "__main__":
    main()
