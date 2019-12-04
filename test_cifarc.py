import numpy as np
import os
import argparse
import torch
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from models.wrn_with_pen import WideResNet
from dataset import load_cifar10c_names_and_label

parser = argparse.ArgumentParser(description='Trains a CIFAR Classifier',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--cifarc_dir', type=str, default='~/workspace/CIFAR-10-C', help='CIFAR10-C dataset dir')
parser.add_argument('--model', '-m', type=str, default='wrn',
                    choices=['allconv', 'wrn'], help='Choose architecture.')
# Optimization options
parser.add_argument('--epochs', '-e', type=int, default=100, help='Number of epochs to train.')
parser.add_argument('--learning_rate', '-lr', type=float, default=0.1, help='The initial learning rate.')
parser.add_argument('--batch_size', '-b', type=int, default=128, help='Batch size.')
parser.add_argument('--test_bs', type=int, default=200)
parser.add_argument('--momentum', type=float, default=0.9, help='Momentum.')
parser.add_argument('--decay', '-d', type=float, default=0.0005, help='Weight decay (L2 penalty).')
# WRN Architecture
parser.add_argument('--layers', default=40, type=int, help='total number of layers')
parser.add_argument('--widen-factor', default=2, type=int, help='widen factor')
parser.add_argument('--droprate', default=0.0, type=float, help='dropout probability')
# Checkpoints
parser.add_argument('--load', '-l', type=str, default='./snapshots/tune',
                    help='Checkpoint path to resume / test.')
parser.add_argument('--test', '-t', action='store_true', help='Test only flag.')
# Acceleration
parser.add_argument('--ngpu', type=int, default=1, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
args = parser.parse_args()

state = {k: v for k, v in args._get_kwargs()}
print(state)


def load_network(load_path, dataset, network):
    for i in range(102 - 1, -1, -1):
        model_name = os.path.join(load_path, dataset + '_baseline_epoch_' + str(i) + '.pt')
        if os.path.isfile(model_name):
            network.load_state_dict(torch.load(model_name), strict=False)
            print('Model restored! Epoch:', i)
            return

    assert False, "could not resume"


def evaluate(net, filename, tensor_y):
    with open(filename, 'rb') as f:
        x = np.load(f)
    x_list = [x[100:10000], x[10100:20000], x[20100:30000], x[30100:40000], x[40100:50000]]
    x = np.concatenate(x_list, axis=0)
    x = x.astype(np.float32) / 255.
    x = np.rollaxis(x, 3, 1)

    tensor_x = torch.from_numpy(x).float()
    testset = torch.utils.data.TensorDataset(tensor_x, tensor_y)
    test_loader = torch.utils.data.DataLoader(testset, batch_size=200, shuffle=False, num_workers=4)

    net.eval()
    torch.set_grad_enabled(False)
    loss, acc, count = 0, 0, 0
    for i, batch in enumerate(test_loader):
        bx = batch[0].cuda()
        by = batch[1].cuda()

        count += by.size(0)

        adv_bx = bx
        with torch.no_grad():
            logits = net(adv_bx * 2 - 1)[0]

        local_loss = F.cross_entropy(logits.data, by, reduction='sum')
        loss += local_loss.cpu().data.numpy()
        acc += (torch.max(logits, dim=1)[1] == by).float().sum(0).cpu().data.numpy()
    loss /= count
    acc /= count

    torch.set_grad_enabled(True)
    return loss, acc


def main():
    torch.manual_seed(1)
    np.random.seed(1)

    names, y = load_cifar10c_names_and_label(args.cifarc_dir)

    y_list = [y[100:10000], y[10100:20000], y[20100:30000], y[30100:40000], y[40100:50000]]
    y = np.concatenate(y_list, axis=0)
    tensor_y = torch.from_numpy(y).long()

    num_classes = 10

    # Create model
    net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate)

    # Restore model if desired
    if args.load != '':
        load_network(args.load, 'cifar10', net)

    # Set CUDA
    if args.ngpu > 0:
        net.cuda()
        torch.cuda.manual_seed(1)

    cudnn.benchmark = True

    tot = 0.
    for name in names:
        loss, acc = evaluate(net, name, tensor_y)
        print('\n{} Test Acc: {:.4f}'.format(name, acc))
        tot += acc
    print('Total : {:.4f}'.format(tot / len(names)))


if __name__ == '__main__':
    main()
