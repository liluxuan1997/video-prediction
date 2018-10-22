# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx
import torch.optim.lr_scheduler as lr_sch
import torch.utils.data as Data

from ConvLSTMnet import ConvLSTM
from CausualLSTMnet import CausualLSTM
from disNet import disNet
from tensorboard_logging import Logger
from data_loader import InputHandle
from metrics import cross_entropy,psnr
#os.environ['CUDA_VISIBLE_DEVICES']="1,2,3"

parser = argparse.ArgumentParser(description='Language Model')
parser.add_argument('-p', '--path', type=str, default='../data/moving-mnist-example',
                    help='path to dataset')
parser.add_argument('--nhid', type=int, default=(64,64,64,64),
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=4,
                    help='number of layers')
parser.add_argument('--lr_g', type=float, default=1e-6,
                    help='initial learning rate')
parser.add_argument('--lr_d', type=float, default=1e-5,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.3,
                    help='gradient clipping')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
parser.add_argument('--weight-decay', '--wd', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)')

parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=8, metavar='N',
                    help='batch size')
parser.add_argument('--batch_size_d', type=int, default=16, metavar='N',
                    help='batch size for d')
parser.add_argument('--kernel_size',default=(5,5),
                    help='kernel size of convLSTM')
parser.add_argument('--seq_length',default=20,
                    help='total input and output length')
parser.add_argument('--input_length',default=10,
                    help='input length of a clip')

parser.add_argument('--save_g', type=str, default='model_generator_new_pretrain.pth',
                    help='path to save the generator model')
parser.add_argument('--save_d', type=str, default='model_discriminator_new_pretrain.pth',
                    help='path to save the discriminator model')
parser.add_argument('-t', '--test', default='', type=str, metavar='PATH',
                   help='path to save test data')

parser.add_argument('--pretrain', default='model_best.pth', type=str, metavar='PATH',
                   help='path to pretrianed model')


batch_size_val = 5
seed = 1111
IMG_HEIGHT = 64
IMG_WIDTH = 64
IMG_CHANNELS = 1

def main():
    global args
    args = parser.parse_args()
    
    torch.manual_seed(seed)
    torch.cuda.set_device(0)

    #build model
    g = ConvLSTM((IMG_HEIGHT, IMG_WIDTH), IMG_CHANNELS, args.nhid, args.kernel_size, args.nlayers,
                     True, args.seq_length, args.input_length).cuda()

    d = disNet().cuda()

    if args.pretrain:
        with open(args.pretrain, 'rb') as f:
            g = torch.load(f)

    g = torch.nn.DataParallel(g, device_ids=[0, 2, 3]).cuda()
    d = torch.nn.DataParallel(d, device_ids=[0, 2, 3]).cuda()




    l2_loss = nn.MSELoss(size_average=False).cuda()
    l1_loss = nn.L1Loss(size_average=False).cuda()
    criterion = (l1_loss,l2_loss)

    optimizer_g = torch.optim.Adamax(g.parameters(), lr = args.lr_g, betas=(0.9, 0.999))
    optimizer_d = torch.optim.SGD(d.parameters(), args.lr_d,
                                  momentum=args.momentum,
                                  weight_decay=args.weight_decay)
    scheduler = lr_sch.ReduceLROnPlateau(optimizer_d, 'min')

    #load data
    if args.test:
        test_input_param = {'path': os.path.join(args.path,'moving-mnist-test.npz'),
                            'minibatch_size': args.batch_size,
                            'input_data_type': 'float32',
                            'is_output_sequence': True}
        test_input_handle = InputHandle(test_input_param)
        test_input_handle.begin(do_shuffle = False)
        # load the model.
        with open(args.save_g, 'rb') as f:
            g = torch.load(f)
        #run the g on test data.
        test_mae, test_mse, test_psnr = evaluate(g,test_input_handle,criterion)
        print('=' * 120)
        print('| test mae {:5.2f} | test mse {:5.2f} | test psnr {:5.2f}|'.format(
            test_mae, test_mse, test_psnr))
        print('=' * 120)
        return   
    
    logger = Logger(os.path.join('./log','convLSTM_GAN_new_pretrain'))
    train_input_param = {'path': os.path.join(args.path,'moving-mnist-train.npz'),
                            'minibatch_size': args.batch_size,
                            'input_data_type': 'float32',
                            'is_output_sequence': True}
    train_input_handle = InputHandle(train_input_param)

    valid_input_param = {'path': os.path.join(args.path,'moving-mnist-valid.npz'),
                            'minibatch_size': args.batch_size,
                            'input_data_type': 'float32',
                            'is_output_sequence': True}
    valid_input_handle = InputHandle(valid_input_param)

    best_val_loss = None

    #test for evaluate function

    #valid_mae,valid_mse,valid_psnr = evaluate(g,valid_input_handle,criterion)

    for epoch in range(1, args.epochs+1):
        train_input_handle.begin(do_shuffle = True)
        epoch_start_time = time.time()
        train_loss_g, train_loss_d, train_mae, train_mse = train(g,d,train_input_handle,criterion,optimizer_g,optimizer_d,scheduler,epoch,logger)
        valid_mae,valid_mse,valid_psnr = evaluate(g,valid_input_handle,criterion)
        print('\n| end of epoch {:3d} | time: {:5.5f}s | valid mae {:5.2f} |' 
            ' valid mse {:5.2f} | valid psnr {:5.2f}'
                .format(epoch, (time.time() - epoch_start_time),valid_mae,valid_mse,valid_psnr))
        print('-' * 120)

        logger.log_scalar('train_mae',train_mae, epoch)
        logger.log_scalar('train_mse',train_mse, epoch)
        logger.log_scalar('valid_mae',valid_mae,epoch)
        logger.log_scalar('valid_mse',valid_mse, epoch)
        logger.log_scalar('valid_psnr',valid_psnr, epoch)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or valid_mae+valid_mse < best_val_loss:
            with open(args.save_g, 'wb') as f:
                torch.save(g, f)
            with open(args.save_d, 'wb') as f:
                torch.save(g, f)
            best_val_loss = valid_mae+valid_mse

def train(g,d,data_handler,criterion,optimizer_g,optimizer_d,scheduler,epoch,logger):
    # Turn on training mode which enables dropout.
    batch_time = AverageMeter()
    losses_g = AverageMeter()
    losses_d = AverageMeter()
    l1_losses = AverageMeter()
    l2_losses = AverageMeter()
    end_time = time.time()
    g.train()
    d.train()
    l1_loss,l2_loss = criterion
    output_length = args.seq_length-args.input_length
    
    i = 0
    while not data_handler.no_batch_left():
        inputs = data_handler.get_batch()
        data_handler.next()
        real_imgs = inputs[:,args.input_length:args.seq_length,:,:,:]
        pred_imgs = g(inputs)
        
        l1_loss_now = l1_loss(pred_imgs, real_imgs)/(args.batch_size* output_length)
        l2_loss_now = l2_loss(pred_imgs, real_imgs)/(args.batch_size* output_length)

        sample_d_num = args.batch_size * args.seq_length/2
        real_imgs = torch.reshape(real_imgs,(sample_d_num,pred_imgs.shape[2],pred_imgs.shape[3],pred_imgs.shape[4]))
        pred_imgs = torch.reshape(pred_imgs,real_imgs.shape)

        #update d
        prob_real = d(real_imgs)
        prob_pred = d(pred_imgs)

        loss_d = torch.mean(-torch.log(prob_real)) + torch.mean(-torch.log(1-prob_pred))
        optimizer_d.zero_grad()
        loss_d.backward(retain_graph=True)
        optimizer_d.step()


        #update g
        loss_g = torch.mean(-torch.log(prob_pred))+l2_loss_now + l1_loss_now
        optimizer_g.zero_grad()
        loss_g.backward()
        optimizer_g.step()

        loss = l1_loss_now + l2_loss_now
                       
                
        #update batch time and average loss
        batch_time.update(time.time()-end_time)
        end_time = time.time()
        losses_g.update(loss_g.item())
        losses_d.update(loss_d.item())
        l1_losses.update(l1_loss_now.item())
        l2_losses.update(l2_loss_now.item())

        if i%100==0:
            logger.log_scalar('train_loss_g',losses_g.avg, epoch*data_handler.total_batches()+i)
            logger.log_scalar('train_loss_d',losses_d.avg, epoch*data_handler.total_batches()+i)
            scheduler.step(l1_losses.avg+l2_losses.avg) 
        
        bar(i+1, data_handler.total_batches(), "Epoch: {:3d} | ".format(epoch),
            ' | time {batch_time.val:.3f} {batch_time.avg:.3f}  '
            '| loss_g {loss_g.val:2.5f} {loss_g.avg:2.5f} |'
            '| loss_d {loss_d.val:2.5f} {loss_d.avg:2.5f} |'
            '| mae {l1_loss.val:5.2f} {l1_loss.avg:5.2f} |'
            '| mse {l2_loss.val:5.2f} {l2_loss.avg:5.2f} |'.format(
                batch_time=batch_time, loss_g=losses_g, loss_d=losses_d, l1_loss=l1_losses, l2_loss=l2_losses), end_string="")
        i += 1
    return losses_g.avg, losses_d.avg, l1_losses.avg,l2_losses.avg

def evaluate(g,data_handler,criterion):
    # Turn on evaluation mode which disables dropout.
    g.eval()
    total_mse = 0.
    total_mae = 0.
    total_psnr = 0.
    l1_loss,l2_loss = criterion
    output_length = args.seq_length-args.input_length

    data_handler.begin(do_shuffle = False)
    with torch.no_grad():
        while not data_handler.no_batch_left():
            inputs = data_handler.get_batch()
            data_handler.next()
            targets = inputs[:,args.input_length:args.seq_length,:,:,:]
            outputs = g(inputs)
            mae = l1_loss(outputs, targets)/(args.batch_size* output_length)
            mse = l2_loss(outputs, targets)/(args.batch_size* output_length)
            psnr_now = psnr(outputs, targets)/(args.batch_size* output_length)
            total_mae += mae.item()
            total_mse += mse.item()
            total_psnr += psnr_now.item()
    n = data_handler.total_batches()
    return total_mae/n, total_mse/n, total_psnr/n

def bar(current, total, prefix="", suffix="", bar_sz=25, end_string=None):
    sp = ""
    print("\x1b[2K\r", end='')
    for i in range(bar_sz):
        if current * bar_sz // total > i:
            sp += '='
        elif current * bar_sz // total == i:
            sp += '>'
        else:
            sp += ' '
    if current == total:
        if end_string is None:
            print("\r%s[%s]%s" % (prefix, sp, suffix))
        else:
            if end_string != "":
                print("\r%s" % end_string)
            else:
                print("\r", end='')
    else:
        print("\r%s[%s]%s" % (prefix, sp, suffix), end='')
        
class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
 
class myDataset(Data.Dataset):
    def __init__(self, imgs, labels, class_num):
        self.imgs = imgs
        self.labels = labels

    def __getitem__(self, index):
        sample = self.imgs[index]
        target = self.labels[index]

    def __len__(self):
        return len(self.imgs)


if __name__ == '__main__':
    main()
