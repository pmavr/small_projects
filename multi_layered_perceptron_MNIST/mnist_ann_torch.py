from tqdm import tqdm
import torch
from time import time
from torch import nn, optim
from torch.utils.data import DataLoader
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()


def fit_model(model, optimizer, criterion, train_data, val_data, num_epochs, batch_len, dev,
              scheduler = None, silent = False):
    print('======= FITTING MODEL ===========')
    print(model)
    print(optimizer)
    print(criterion)
    print('Batch size: ', batch_len)

    train_loader = DataLoader(train_data, batch_size = batch_len, shuffle = True)
    val_loader = DataLoader(val_data, batch_size = len(val_data))

    hist = { "train_loss": [], "train_acc": [], "val_loss": [], "val_acc": [] }

    print('Train on {} samples, validate on {} samples:'.format(
        len(train_data),
        len(val_data)
    ))

    start_time = time()

    for e in range(num_epochs):
        epoch_train_acc = epoch_train_loss = 0

        model.train()
        for train_images, train_labels in train_loader:
            train_images = train_images.reshape(-1, 28*28).to(dev)
            train_labels = train_labels.to(dev)

            optimizer.zero_grad()
            output = model(train_images)

            cost = criterion(output, train_labels)
            cost.backward()
            optimizer.step()

            ps = torch.exp(output)
            pred_prob, pred_label  = ps.topk(1, dim=1)
            true_label = train_labels.view(*pred_label.shape)
            equals = true_label == pred_label

            epoch_train_acc += torch.mean(equals.type(torch.FloatTensor)).item() / len(train_loader)
            epoch_train_loss += cost.item() / len(train_loader)

        model.eval()
        with torch.no_grad():
            val_images, val_labels = next(iter(val_loader))
            val_images = val_images.reshape(-1, 28*28).to(dev)
            val_labels = val_labels.to(dev)

            logps = model(val_images)
            cost = criterion(logps, val_labels)
            ps = torch.exp(logps)
            pred_prob, pred_label  = ps.topk(1, dim=1)
            true_label = val_labels.view(*pred_label.shape)
            equals = true_label == pred_label

            epoch_val_acc = torch.mean(equals.type(torch.FloatTensor)).item()
            epoch_val_loss = cost.item()

        if scheduler:
            scheduler.step()

        # Hold acuracy and loss scores for current epoch
        hist['train_acc'].append(epoch_train_acc)
        hist['train_loss'].append(epoch_train_loss)
        hist['val_acc'].append(epoch_val_acc)
        hist['val_loss'].append(epoch_val_loss)
        if not silent:
            print('\nEpoch {}/{}: Train Accuracy: {:.4f} - Train Loss: {:.4f} | Val. Accuracy: {:.4f} - Val. Loss: {:.4f}'.format(
                e+1,
                num_epochs,
                epoch_train_acc,
                epoch_train_loss,
                epoch_val_acc,
                epoch_val_loss
            ))
        else:
            print('.', end='')

    training_time = time() - start_time
    print('\nModel fit in : {:.2f}s'.format(training_time))

    return model, hist

def test_model(model, test_data, dev):
    val_loader = DataLoader(test_data, batch_size = len(test_data))
    model.eval()
    with torch.no_grad():
        val_images, val_labels = next(iter(val_loader))
        val_images = val_images.reshape(-1, 28*28).to(dev)
        val_labels = val_labels.to(dev)

        logps = model(val_images)
        ps = torch.exp(logps)
        pred_prob, pred_label  = ps.topk(1, dim=1)
        true_label = val_labels.view(*pred_label.shape)
        equals = true_label == pred_label

        accuracy = torch.mean(equals.type(torch.FloatTensor)).item()

    print('Test on {} samples - Accuracy: {:.4f}'.format(len(test_data), accuracy))

def init_weights_normal(layer):
    if type(layer) == nn.Linear:
        nn.init.normal_(layer.weight, std=.01, mean=.0)

def init_weights_xavier(layer):
    if type(layer) == nn.Linear:
        torch.nn.init.xavier_normal_(layer.weight)

def init_weights_zeros(layer):
    if type(layer) == nn.Linear:
        torch.nn.init.zeros_(layer.weight)

def plot_results(e, h, info):

    epochs_range = [i for i in range(e)]

    fig, (loss_plot, acc_plot) = plt.subplots(1, 2, figsize =(12,4))

    loss_plot.plot(epochs_range, h['train_loss'], color='red', label='train loss')
    loss_plot.plot(epochs_range, h['val_loss'], color='green', label='val loss')
    loss_plot.set_title('Epochs - Loss / {}'.format(info))
    loss_plot.legend()

    acc_plot.plot(epochs_range, h['train_acc'], color='red', label='train acc')
    acc_plot.plot(epochs_range, h['val_acc'], color='green', label='val acc')
    acc_plot.set_title('Epochs - Accuracy / {}'.format(info))
    acc_plot.legend()

    plt.show()


if __name__ == '__main__':

    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")

    train_dataset = torchvision.datasets.MNIST(root='../../data',
                                               train=True,
                                               transform=transforms.ToTensor(),
                                               download=True)

    train_dataset, val_dataset = torch.utils.data.random_split(train_dataset, [50000, 10000])

    test_dataset = torchvision.datasets.MNIST(root='../../data',
                                              train=False,
                                              transform=transforms.ToTensor())

    epochs = 12
    batch_size = 256
    # learning_rate = 0.3
    learning_rate = .0015
    learning_rate_decay = 0.43

    input_layer = nn.Linear(784, 1280)
    nn.init.kaiming_normal_(input_layer.weight)

    hidden1_layer = nn.Linear(1280, 1280)
    nn.init.kaiming_normal_(hidden1_layer.weight)

    hidden2_layer = nn.Linear(1280, 1280)
    nn.init.kaiming_normal_(hidden2_layer.weight)

    output_layer = nn.Linear(1280, 10)
    nn.init.normal_(output_layer.weight, std=0.01)

    network = nn.Sequential(

        input_layer,
        nn.Dropout(0.2),
        nn.LeakyReLU(),

        hidden1_layer,
        nn.Dropout(0.2),
        nn.LeakyReLU(),

        hidden2_layer,
        nn.Dropout(0.1),
        nn.LeakyReLU(),

        output_layer,
        nn.LogSoftmax(dim=1)
    ).to(device)

    loss_func = nn.NLLLoss()
    optim_func = optim.Adam(network.parameters(), lr=learning_rate, )
    # optim_func = optim.SGD(network.parameters(), lr=learning_rate, momentum=0.9)
    sched_func = optim.lr_scheduler.MultiStepLR(optim_func, milestones=[5, 7, 9], gamma=learning_rate_decay)

    network, history = fit_model(
        model=network,
        optimizer=optim_func,
        criterion=loss_func,
        train_data=train_dataset,
        val_data=val_dataset,
        num_epochs=epochs,
        batch_len=batch_size,
        dev=device,
        scheduler=sched_func
    )

    test_model(
        model=network,
        test_data=test_dataset,
        dev=device)

    plot_results(epochs, history, info='Adam | 784-1280-1280-1280-10')