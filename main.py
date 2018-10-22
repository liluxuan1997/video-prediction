# coding: utf-8
import argparse
import time
import math
import os
import torch
import torch.nn as nn
import torch.onnx
from skimage.io import imsave
from pathlib import Path
import numpy as np

from ConvLSTMnet import ConvLSTM
from CausualLSTMnet import CausualLSTM
from tensorboard_logging import Logger
from data_loader import InputHandle
from metrics import cross_entropy,psnr
#os.environ['CUDA_VISIBLE_DEVICES']="1,2,3"

parser = argparse.ArgumentParser(description='Language Model')
parser.add_argument('-p', '--path', type=str, default='../data/moving-mnist-example',
                    help='path to dataset')
parser.add_argument('--nhid', type=int, default=(128,64,64,64),
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=4,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=1e-3,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.3,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=4, metavar='N',
                    help='batch size')
parser.add_argument('--kernel_size',default=(5,5),
                    help='kernel size of convLSTM')
parser.add_argument('--seq_length',default=20,
                    help='total input and output length')
parser.add_argument('--input_length',default=10,
                    help='input length of a clip')

parser.add_argument('--dropout', type=float, default=0.65,
                    help='dropout applied to layers')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str, default='model_best.pth',
                    help='path to save the final model')
parser.add_argument('-t', '--test', default='', type=str, metavar='PATH',
                   help='path to save test data')


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
    model = CausualLSTM((IMG_HEIGHT, IMG_WIDTH), IMG_CHANNELS, args.nhid, args.kernel_size, args.nlayers,
                     True, args.seq_length, args.input_length).cuda()

    model = torch.nn.DataParallel(model, device_ids=[0, 2, 3]).cuda()

    l2_loss = nn.MSELoss(size_average=False).cuda()
    l1_loss = nn.L1Loss(size_average=False).cuda()
    criterion = (l1_loss,l2_loss)

    if args.test:
        test_input_param = {'path': os.path.join(args.path,'moving-mnist-test.npz'),
                            'minibatch_size': args.batch_size,
                            'input_data_type': 'float32',
                            'is_output_sequence': True}
        test_input_handle = InputHandle(test_input_param)
        test_input_handle.begin(do_shuffle = False)
        # load the model.
        with open(args.save, 'rb') as f:
            model = torch.load(f)
        #run the model on test data.
        test_mae, test_mse, test_psnr = evaluate(model,test_input_handle,criterion)
        print('=' * 120)
        print('| test mae {:5.2f} | test mse {:5.2f} | test psnr {:5.2f}|'.format(
            test_mae, test_mse, test_psnr))
        print('=' * 120)
        return 

    optimizer = torch.optim.Adamax(model.parameters(), lr = args.lr, betas=(0.9, 0.999))
    #load data
  
    
    logger = Logger(os.path.join('./log','convLSTM'))
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

    valid_mae,valid_mse,valid_psnr = evaluate(model,valid_input_handle,criterion)

    for epoch in range(1, args.epochs+1):
        train_input_handle.begin(do_shuffle = True)
        epoch_start_time = time.time()
        train_loss, train_mae, train_mse = train(model,train_input_handle,criterion,optimizer,epoch)
        valid_mae,valid_mse,valid_psnr = evaluate(model,valid_input_handle,criterion)
        print('\n| end of epoch {:3d} | time: {:5.5f}s | valid mae {:5.2f} |' 
            ' valid mse {:5.2f} | valid psnr {:5.2f}'
                .format(epoch, (time.time() - epoch_start_time),valid_mae,valid_mse,valid_psnr))
        print('-' * 120)
        logger.log_scalar('train_loss',train_loss, epoch)
        logger.log_scalar('train_mae',train_mae, epoch)
        logger.log_scalar('train_mse',train_mse, epoch)
        logger.log_scalar('valid_mae',valid_mae,epoch)
        logger.log_scalar('valid_mse',valid_mse, epoch)
        logger.log_scalar('valid_psnr',valid_psnr, epoch)
        # Save the model if the validation loss is the best we've seen so far.
        if not best_val_loss or valid_mae+valid_mse < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(model, f)
            best_val_loss = valid_mae+valid_mse

def train(model,data_handler,criterion,optimizer,epoch):
    # Turn on training mode which enables dropout.
    batch_time = AverageMeter()
    losses = AverageMeter()
    l1_losses = AverageMeter()
    l2_losses = AverageMeter()
    end_time = time.time()
    model.train()
    l1_loss,l2_loss = criterion
    output_length = args.seq_length-args.input_length
    
    i = 0
    while not data_handler.no_batch_left():
        inputs = data_handler.get_batch()
        data_handler.next()
        targets = inputs[:,args.input_length:args.seq_length,:,:,:]
        model.zero_grad()
        outputs = model(inputs)
        l1_loss_now = l1_loss(outputs, targets)/(args.batch_size* output_length)
        l2_loss_now = l2_loss(outputs, targets)/(args.batch_size* output_length)
        loss = l1_loss_now + l2_loss_now
        loss.backward()
        optimizer.step()
                    
        #update batch time and average loss
        batch_time.update(time.time()-end_time)
        end_time = time.time()
        losses.update(loss.item())
        l1_losses.update(l1_loss_now.item())
        l2_losses.update(l2_loss_now.item())
        
        bar(i+1, data_handler.total_batches(), "Epoch: {:3d} | ".format(epoch),
            ' | time {batch_time.val:.3f} {batch_time.avg:.3f}  '
            '| loss {loss.val:5.2f} {loss.avg:5.2f} |'
            '| mae {l1_loss.val:5.2f} {l1_loss.avg:5.2f} |'
            '| mse {l2_loss.val:5.2f} {l2_loss.avg:5.2f} |'.format(
                batch_time=batch_time, loss=losses, l1_loss=l1_losses, l2_loss=l2_losses), end_string="")
        i += 1
    return losses.avg,l1_losses.avg,l2_losses.avg

def evaluate(model,data_handler,criterion):
    # Turn on evaluation mode which disables dropout.
    model.eval()
    total_mse = 0.
    total_mae = 0.
    total_psnr = 0.
    l1_loss,l2_loss = criterion
    output_length = args.seq_length-args.input_length
    
    if args.test:
        i = 0
        pred_dir = os.path.join(args.test,'pred_imgs')
        gt_dir = os.path.join(args.test,'gt_imgs')
        if not Path(pred_dir).exists():
            os.mkdir(pred_dir)
        if not Path(gt_dir).exists():
            os.mkdir(gt_dir)

    data_handler.begin(do_shuffle = False)
    with torch.no_grad():
        while not data_handler.no_batch_left():
            inputs = data_handler.get_batch()
            data_handler.next()
            targets = inputs[:,args.input_length:args.seq_length,:,:,:]
            model.zero_grad()
            outputs = model(inputs)
            mae = l1_loss(outputs, targets)/(args.batch_size* output_length)
            mse = l2_loss(outputs, targets)/(args.batch_size* output_length)
            psnr_now = psnr(outputs, targets)/(args.batch_size* output_length)
            total_mae += mae.item()
            total_mse += mse.item()
            total_psnr += psnr_now.item()
            print('outputs max pixel:'+str(torch.max(outputs)))
            print('targets max pixel:'+str(torch.max(targets)))
            outputs = outputs /torch.max(outputs)
            if args.test and i <= 20:
                targets = targets.cpu()
                outputs = outputs.cpu()
                for k in range(args.batch_size):
                    img_pred_dir = os.path.join(pred_dir,'{:5d}'.format(i))
                    img_gt_dir = os.path.join(gt_dir,'{:5d}'.format(i))
                    if not Path(img_pred_dir).exists():
                        os.mkdir(img_pred_dir)
                    if not Path(img_gt_dir).exists():
                        os.mkdir(img_gt_dir)
                    i += 1
                    for j in range(output_length):
                        img_pred = outputs[k,j,:,:,:].numpy()
                        img_pred = np.transpose(img_pred,(1,2,0))
                        img_pred = np.reshape(img_pred,(64,64))
                        img_pred_name = os.path.join(img_pred_dir,'{:5d}.jpg'.format(j))
                        imsave(img_pred_name,img_pred)
                        img_gt = targets[k,j,:,:,:].numpy()
                        img_gt = np.transpose(img_gt,(1,2,0))
                        img_gt = np.reshape(img_gt,(64,64))
                        img_gt_name = os.path.join(img_gt_dir,'{:5d}.jpg'.format(j+10))
                        imsave(img_gt_name,img_gt)
                        img_input = inputs[k,j,:,:,:].cpu().numpy()
                        img_input = np.transpose(img_input,(1,2,0))
                        img_input = np.reshape(img_input,(64,64))
                        img_input_name = os.path.join(img_gt_dir,'{:5d}.jpg'.format(j))
                        imsave(img_input_name,img_input)
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
        

if __name__ == '__main__':
    main()
