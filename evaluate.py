import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets

from helpers.loaders import getdataloader, getwmloader

device = 'cuda' if torch.cuda.is_available() else 'cpu'

def evaluate_accuracy(model_path, dataset, test_db_path):
    net = torch.load(model_path)
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()

    _, testloader, _ = getdataloader(dataset, test_db_path, test_db_path, 100)

    net.eval()
    test_loss = 0
    correct = 0
    total = 0
    with torch.no_grad():
        for batch_idx, (inputs, targets) in enumerate(testloader):
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = net(inputs)
            loss = criterion(outputs, targets)

            test_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += targets.size(0)
            correct += predicted.eq(targets.data).cpu().sum().item()

    accuracy = 100 * correct / total
    print(f'Test Accuracy: {accuracy:.2f}%')
    return accuracy

def evaluate_effectiveness(model_path, wm_path, wm_lbl):
    net = torch.load(model_path)
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()

    wmloader = getwmloader(wm_path, 2, wm_lbl)

    net.eval()
    wm_correct = 0
    wm_total = 0
    with torch.no_grad():
        for wm_idx, (wm_inputs, wm_targets) in enumerate(wmloader):
            wm_inputs, wm_targets = wm_inputs.to(device), wm_targets.to(device)
            wm_outputs = net(wm_inputs)
            _, wm_predicted = torch.max(wm_outputs.data, 1)
            wm_total += wm_targets.size(0)
            wm_correct += wm_predicted.eq(wm_targets.data).cpu().sum().item()

    effectiveness = 100 * wm_correct / wm_total
    print(f'Watermark Effectiveness: {effectiveness:.2f}%')
    return effectiveness

def fine_tune_model(model_path, trainloader, testloader, wmloader, epochs=5):
    net = torch.load(model_path)
    net = net.to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9, weight_decay=5e-4)

    for epoch in range(epochs):
        net.train()
        for batch_idx, (inputs, targets) in enumerate(trainloader):
            inputs, targets = inputs.to(device), targets.to(device)
            optimizer.zero_grad()
            outputs = net(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    # Save the fine-tuned model
    torch.save(net, 'checkpoint/fine_tuned_model.t7')
    return net

# Example usage:
original_accuracy = evaluate_accuracy('checkpoint/original_model.t7', 'cifar10', './data')
watermarked_accuracy = evaluate_accuracy('checkpoint/watermarked_model.t7', 'cifar10', './data')
print(f'Accuracy drop: {original_accuracy - watermarked_accuracy:.2f}%')

effectiveness = evaluate_effectiveness('checkpoint/watermarked_model.t7', './data/trigger_set/', 'labels-cifar.txt')

trainloader, testloader, _ = getdataloader('cifar10', './data', './data', 100)
wmloader = getwmloader('./data/trigger_set/', 2, 'labels-cifar.txt')
fine_tune_model('checkpoint/watermarked_model.t7', trainloader, testloader, wmloader)

fine_tuned_effectiveness = evaluate_effectiveness('checkpoint/fine_tuned_model.t7', './data/trigger_set/', 'labels-cifar.txt')
