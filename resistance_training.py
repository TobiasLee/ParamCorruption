from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torchvision import datasets, transforms
from torch.optim.lr_scheduler import StepLR
from models import LinearModel, ConvNet
from tqdm import tqdm
from attacker import GradAttacker


def train(args, model, device, train_loader, optimizer, epoch):
    model.train()
    for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = model(data.view(data.size(0), -1))
        loss = F.nll_loss(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss.item()))
            if args.dry_run:
                break


def adv_train(args, model, attacker_class, device, train_loader, optimizer, epoch, K=1):
    model.train()
    for batch_idx, (data, target) in tqdm(enumerate(train_loader)):
        # zero grad
        optimizer.zero_grad()
        model.zero_grad()

        param_mp = {}
        grad_mp = {}
        param_dict = dict(model.named_parameters())
        # save a model param copy and grad copy
        for name in param_dict:
            param_mp[name] = param_dict[name].data.clone().detach()
            if param_dict[name].grad is not None:
                grad_mp[name] = param_dict[name].grad.data.clone().detach()
            else:
                grad_mp[name] = torch.zeros_like(param_dict[name].data)

        defence_parameters = model.parameters()
        attacker = attacker_class(defence_parameters, lr=1.5 * args.eps / K, eps=args.eps )

        # move data to device
        data, target = data.to(device), target.to(device)
        data = data.view(data.size(0), -1)
        output = model(data)
        nll_loss = F.nll_loss(output, target) / 4.0 /   ( K + 1.0)
        nll_loss.backward()  # get grad

        for name in param_dict:  # save grad of normal loss
            grad_mp[name].add_(param_dict[name].grad.data)

        loss = nll_loss.item()
        for i in range(K):
            attacker.step()  # attack the model
            model.zero_grad()
            loss_attacked = F.nll_loss(model(data), target) / 4.0   / (K + 1.0)
            loss_attacked.backward()  # get the grad
            for name in param_dict:
                grad_mp[name].add_(param_dict[name].grad.data)  # record attacked grad
            loss += loss_attacked.item()
        for name in param_dict:  # copy param back
            param_dict[name].data = param_mp[name].clone().detach()
            param_dict[name].grad.data = grad_mp[name].clone().detach()
        if batch_idx % 4 == 3:
            optimizer.step()
            optimizer.zero_grad()
 
        if batch_idx % args.log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch, batch_idx * len(data), len(train_loader.dataset),
                       100. * batch_idx / len(train_loader), loss))
            if args.dry_run:
                break


def test(model, device, test_loader):
    model.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = model(data.view(data.size(0), -1))
            test_loss += F.nll_loss(output, target, reduction='sum').item()  # sum up batch loss
            pred = output.argmax(dim=1, keepdim=True)  # get the index of the max log-probability
            correct += pred.eq(target.view_as(pred)).sum().item()

    test_loss /= len(test_loader.dataset)

    print('\nTest set: Average loss: {:.4f}, Accuracy: {}/{} ({:.2f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


def main():
    # Training settings
    parser = argparse.ArgumentParser(description='PyTorch MNIST Example')
    parser.add_argument('--batch-size', type=int, default=64, metavar='N',
                        help='input batch size for training (default: 64)')
    parser.add_argument('--test-batch-size', type=int, default=1000, metavar='N',
                        help='input batch size for testing (default: 1000)')
    parser.add_argument('--epochs', type=int, default=1, metavar='N',
                        help='number of epochs to train (default: 14)')
    parser.add_argument('--lr', type=float, default=1.0, metavar='LR',
                        help='learning rate (default: 1.0)')
    parser.add_argument('--gamma', type=float, default=0.7, metavar='M',
                        help='Learning rate step gamma (default: 0.7)')
    parser.add_argument('--no-cuda', action='store_true', default=False,
                        help='disables CUDA training')
    parser.add_argument('--dry-run', action='store_true', default=False,
                        help='quickly check a single pass')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--adv_train_k', type=int, default=2, metavar='N',
                        help='conduct K steps attack in adv training')
    parser.add_argument('--save-model', action='store_true', default=False,
                        help='For Saving the current Model')
    parser.add_argument('--attack', action='store_true', default=False,
                        help='attack model')
    parser.add_argument('--LP', type=str, default="l2",
                        help='Random Corruption Norm Constrain')
    parser.add_argument('--eps', type=float, default=1e-4,
                        help='Random Corruption Epsilon')
    parser.add_argument('--attack_lr', type=float, default=1e-3,
                        help='Grad based attacker learning rate')
    args = parser.parse_args()
    use_cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    device = torch.device("cuda" if use_cuda else "cpu")

    train_kwargs = {'batch_size': args.batch_size}
    test_kwargs = {'batch_size': args.test_batch_size}
    if use_cuda:
        cuda_kwargs = {'num_workers': 1,
                       'pin_memory': True,
                       'shuffle': True}
        train_kwargs.update(cuda_kwargs)
        test_kwargs.update(cuda_kwargs)

    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    dataset1 = datasets.MNIST('./', train=True, download=True,
                              transform=transform)
    dataset2 = datasets.MNIST('./', train=False,
                              transform=transform)
    train_loader = torch.utils.data.DataLoader(dataset1, **train_kwargs)
    test_loader = torch.utils.data.DataLoader(dataset2, **test_kwargs)

    model = LinearModel().to(device)  # Net().to(device)
    optimizer = optim.Adadelta(model.parameters(), lr=args.lr)

    scheduler = StepLR(optimizer, step_size=1, gamma=args.gamma)
    for epoch in range(1, args.epochs + 1):
        if epoch < 5:
            train(args, model, device, train_loader, optimizer, epoch)
        else:
            print("Start adversarial training")
            # conduct adv training
            adv_train(args, model, GradAttacker, device, train_loader, optimizer, epoch, K=args.adv_train_k)
        test(model, device, test_loader)
        scheduler.step()

    if args.attack:
        print("start attack")
        attacker = GradAttacker(model.parameters(), lr=args.attack_lr, eps=args.eps, LP=args.LP)
        train(args, model, device, train_loader, optimizer=attacker, epoch='attack epoch')
        print("Accuracy After attack:")
        test(model, device, test_loader)

    if args.save_model:
        torch.save(model.state_dict(), "mnist_linear.pt")


if __name__ == '__main__':
    main()
