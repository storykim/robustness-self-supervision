import numpy as np
import os
import argparse
import time
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
from models.wrn_with_pen import WideResNet
from skimage import color
from scipy.ndimage.filters import gaussian_filter
from dataset import get_cifar_dataloader, load_cifar10c_for_validation

parser = argparse.ArgumentParser(description='Trains a CIFAR Classifier',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--dataset', type=str, default='cifar10', choices=['cifar10', 'cifar100'],
                    help='Choose between CIFAR-10, CIFAR-100.')
parser.add_argument('--data_dir', type=str, default='~/data', help='Path to save CIFAR-10(or CIFAR-100) dataset')
parser.add_argument('--cifarc_dir', type=str, default='~/workspace/CIFAR-10-C', help='CIFAR10-C dataset dir')
parser.add_argument('--save', '-s', type=str, default='./snapshots/rot_five', help='Folder to save checkpoints.')
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
# Acceleration
parser.add_argument('--ngpu', type=int, default=2, help='0 = CPU.')
parser.add_argument('--prefetch', type=int, default=4, help='Pre-fetching threads.')
parser.add_argument('--lamb', type=float, default=0.5, help='Lambda.')
parser.add_argument('--lambda_fix_epoch', type=int, default=100, help='Pretraining only with rotation loss')
# Self-supervision
parser.add_argument('--rot', action='store_true', help='Rotation flag')
parser.add_argument('--color', action='store_true', help='Color flag')
parser.add_argument('--blur', action='store_true', help='Blur flag')
parser.add_argument('--encode', action='store_true', help='Encode flag')
parser.add_argument('--inpaint', action='store_true', help='Inpaint flag')
mse = nn.MSELoss()


def cosine_annealing(step, total_steps, lr_max, lr_min):
    return lr_min + (lr_max - lr_min) * 0.5 * (
            1 + np.cos(step / total_steps * np.pi))


def get_params(model):
    return [p for p in model.named_parameters() if p[1].requires_grad]


def set_param(curr_mod, name, param):
    if '.' in name:
        n = name.split('.')
        module_name = n[0]
        rest = '.'.join(n[1:])
        for name, mod in curr_mod.named_children():
            if module_name == name:
                set_param(mod, rest, param)
                break
    else:
        curr_mod._parameters[name] = param


def calc_loss(network, bx, orig_bx, by, additional_by, lamb_list, args):
    # forward
    logits, pen = network(bx * 2 - 1)

    curr_batch_size = len(by)
    loss = F.cross_entropy(logits[:curr_batch_size], by)

    cur_idx = 1
    loss_list = []
    if args.rot:
        rot_pred = network.rot_pred(pen[:4 * curr_batch_size])
        loss_rot = F.cross_entropy(rot_pred, additional_by[0])
        loss_list.append(loss_rot)
        cur_idx += 3

    if args.color:
        color_pred = network.color_pred(
            pen[cur_idx * curr_batch_size:(cur_idx + 1) * curr_batch_size].view(-1, 128, 1, 1))
        color_pred = color_pred.view(-1, 64, 32 * 32).permute(0, 2, 1)
        a_pred = color_pred[..., :32].contiguous().view(-1, 32)
        b_pred = color_pred[..., 32:].contiguous().view(-1, 32)
        color_by = additional_by[0 if args.rot else 1]
        a_label = color_by[..., 0].view(-1)
        b_label = color_by[..., 1].view(-1)

        loss_col = F.cross_entropy(a_pred, a_label) + F.cross_entropy(b_pred, b_label)
        loss_list.append(loss_col)
        cur_idx += 1

    if args.blur:
        blur_pen = pen[cur_idx * curr_batch_size:(cur_idx + 1) * curr_batch_size]
        blur_pred = network.blur_pred(network.blur_init(blur_pen).view(-1, 128, 1, 1))
        loss_blur = 30 * mse(blur_pred, orig_bx)
        loss_list.append(loss_blur)
        cur_idx += 1

    if args.encode:
        encode_pen = pen[:curr_batch_size]
        encode_pred = network.encode_pred(network.encode_init(encode_pen).view(-1, 128, 1, 1))
        loss_encode = 30 * mse(encode_pred, orig_bx)
        loss_list.append(loss_encode)

    if args.inpaint:
        inpaint_pen = pen[cur_idx * curr_batch_size:(cur_idx + 1) * curr_batch_size]
        inpaint_pred = network.inpaint_pred(network.inpaint_init(inpaint_pen).view(-1, 128, 1, 1))
        loss_inpaint = 30 * mse(inpaint_pred, orig_bx[:, :, 12:20, 12:20])
        loss_list.append(loss_inpaint)

    for idx, lamb in enumerate(lamb_list):
        loss += lamb * loss_list[idx]

    return loss


def main():
    args = parser.parse_args()

    state = {k: v for k, v in args._get_kwargs()}
    print(state)

    torch.manual_seed(1)
    np.random.seed(1)

    # Get data
    train_loader, test_loader, num_classes = get_cifar_dataloader(args.dataset, args.data_dir, args.batch_size,
                                                                  args.test_bs, args.prefetch)

    # Create model
    net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate, rot=args.rot,
                     color=args.color, blur=args.blur, encode=args.encode, inpaint=args.inpaint)

    if args.ngpu > 0:
        net.cuda()
        torch.cuda.manual_seed(1)
    cudnn.benchmark = True

    # Set optimizer
    optimizer = torch.optim.SGD(
        net.parameters(), state['learning_rate'], momentum=state['momentum'],
        weight_decay=state['decay'], nesterov=True)

    # Set scheduler
    scheduler = torch.optim.lr_scheduler.LambdaLR(
        optimizer,
        lr_lambda=lambda step: cosine_annealing(
            step,
            args.epochs * len(train_loader),
            1,  # since lr_lambda computes multiplicative factor
            1e-6 / args.learning_rate))

    # Load CIFAR10C dataset for validation
    inf_test_loader = load_cifar10c_for_validation(args.cifarc_dir, args.prefetch)

    # Define variables
    cur_step = 0
    tot_step = args.epochs * len(train_loader)

    num_of_ce = args.rot + args.color
    num_of_mse = args.blur + args.encode + args.inpaint
    lamb = [args.lamb] * (num_of_ce + num_of_mse)

    # /////////////// Training ///////////////
    def train(epoch):
        nonlocal lamb, cur_step
        net.train()  # enter train mode
        loss_avg = 0.0
        for bx, by in train_loader:
            cur_step += 1
            scheduler.step()

            curr_batch_size = bx.size(0)
            orig_bx = bx
            bx = bx.numpy()
            additional_by = []
            additional_bx = []

            # Make x and y for each of self-supervison tasks
            if args.rot:
                by_prime = torch.cat((torch.zeros(curr_batch_size), torch.ones(curr_batch_size),
                                      2 * torch.ones(curr_batch_size), 3 * torch.ones(curr_batch_size)), 0).long()
                additional_by.append(by_prime)
                additional_bx += [np.rot90(bx, 1, axes=(2, 3)), np.rot90(bx, 2, axes=(2, 3)),
                                  np.rot90(bx, 3, axes=(2, 3))]

            cl_bx = None
            if args.color:
                cl_bx = np.moveaxis(bx, 1, 3)
                gray = np.dot(cl_bx, [0.2989, 0.5870, 0.1140])
                gray = np.stack((gray,) * 3, axis=-1)
                color_by = color.rgb2lab(cl_bx)[..., 1:].reshape(-1, 32 * 32, 2).astype(np.int)
                color_by = color_by + 128
                color_by //= 8
                color_by = torch.LongTensor(color_by)
                additional_by.append(color_by)
                additional_bx.append(np.moveaxis(gray, 3, 1))

            if args.blur:
                if cl_bx is None: cl_bx = np.moveaxis(bx, 1, 3)
                additional_bx.append(np.moveaxis(gaussian_filter(cl_bx, sigma=7. / 255), 3, 1))
                # No need to add label

            if args.encode:
                # Nothing to do
                pass

            if args.inpaint:
                if cl_bx is None: cl_bx = np.moveaxis(bx, 1, 3)
                inpaint_input = np.copy(cl_bx)
                inpaint_input[:, 12:20, 12:20, :] = .9999
                additional_bx.append(np.moveaxis(inpaint_input, 3, 1))

            bx = np.concatenate([bx] + additional_bx)
            bx = torch.FloatTensor(bx)

            bx, orig_bx, by = bx.cuda(), orig_bx.cuda(), by.cuda()
            additional_by = [aby.cuda() for aby in additional_by]

            if args.lambda_fix_epoch >= epoch:
                # Training without Auto-ML
                loss = calc_loss(net, bx, orig_bx, by, additional_by, lamb, args)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()

                # exponential moving average
                loss_avg = loss_avg * 0.9 + float(loss) * 0.1
                continue

            # Change lambda to tensor
            lambda_tensor = torch.tensor(lamb).requires_grad_(True).cuda()

            # Copy original network
            meta_net = WideResNet(args.layers, num_classes, args.widen_factor, dropRate=args.droprate, rot=args.rot,
                                  color=args.color, blur=args.blur, encode=args.encode, inpaint=args.inpaint)
            meta_net.load_state_dict(net.state_dict())
            meta_net.cuda()

            # Calc loss for meta network
            loss = calc_loss(meta_net, bx, orig_bx, by, additional_by, lambda_tensor, args)

            # Update parameters temporary
            meta_net.zero_grad()
            param_name_pairs = get_params(meta_net)
            params = [param for name, param in param_name_pairs]
            grads = torch.autograd.grad(loss, params, create_graph=True, only_inputs=True)
            alpha = cosine_annealing(cur_step, tot_step, 1, 1e-6 / args.learning_rate)
            beta = 0.005 * 2

            for (name, param), g in zip(param_name_pairs, grads):
                tmp = param - alpha * g
                set_param(meta_net, name, tmp)

            lambda_tensor.requires_grad_()

            # Second forward pass
            x_val, y_val = next(inf_test_loader)
            x_val, y_val = x_val.cuda(), y_val.cuda()
            logits2, _ = meta_net(x_val * 2 - 1)
            loss2 = F.cross_entropy(logits2, y_val)

            grad_lambda = torch.autograd.grad(loss2, lambda_tensor, only_inputs=True)[0]
            new_lambda = lambda_tensor - beta * grad_lambda
            new_lambda = torch.clamp(new_lambda, min=0)
            lamb = new_lambda.tolist()

            # Update original network's parameters
            loss = calc_loss(net, bx, orig_bx, by, additional_by, lamb, args)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            # exponential moving average
            loss_avg = loss_avg * 0.9 + float(loss) * 0.1

        state['train_loss'] = loss_avg

    # test function
    def evaluate():
        net.eval()
        loss_avg = 0.0
        correct = 0
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.cuda(), target.cuda()

                # forward
                output, _ = net(data * 2 - 1)
                loss = F.cross_entropy(output, target)

                # accuracy
                pred = output.data.max(1)[1]
                correct += pred.eq(target.data).sum().item()

                # test loss average
                loss_avg += float(loss.data)

        state['test_loss'] = loss_avg / len(test_loader)
        state['test_accuracy'] = correct / len(test_loader.dataset)

    # Make save directory
    if not os.path.exists(args.save):
        os.makedirs(args.save)
    if not os.path.isdir(args.save):
        raise Exception('%s is not a dir' % args.save)

    with open(os.path.join(args.save, args.dataset +
                                      '_baseline_training_results.csv'), 'w') as f:
        f.write('epoch,time(s),train_loss,test_loss,test_error(%)\n')

    print('Beginning Training\n')

    # Main loop
    for epoch in range(args.epochs):
        state['epoch'] = epoch

        begin_epoch = time.time()

        train(epoch)
        evaluate()

        # Save model
        torch.save(net.state_dict(),
                   os.path.join(args.save, args.dataset +
                                '_baseline_epoch_' + str(epoch) + '.pt'))
        # Let us not waste space and delete the previous model
        prev_path = os.path.join(args.save, args.dataset +
                                 '_baseline_epoch_' + str(epoch - 1) + '.pt')
        if os.path.exists(prev_path): os.remove(prev_path)

        # Store & show results
        with open(os.path.join(args.save, args.dataset +
                                          '_baseline_training_results.csv'), 'a') as f:
            f.write('%03d,%05d,%0.6f,%0.5f,%0.2f,%s\n' % (
                (epoch + 1),
                time.time() - begin_epoch,
                state['train_loss'],
                state['test_loss'],
                100 - 100. * state['test_accuracy'],
                ' / '.join("%.4f" % x for x in lamb),
            ))

        print(
            'Epoch {0:3d} | Time {1:5d} | Train Loss {2:.4f} | Test Loss {3:.3f} | Test Error {4:.2f} Lambda {5:s}'.format(
                (epoch + 1),
                int(time.time() - begin_epoch),
                state['train_loss'],
                state['test_loss'],
                100 - 100. * state['test_accuracy'],
                ' / '.join("%.4f" % x for x in lamb))
        )


if __name__ == "__main__":
    main()
