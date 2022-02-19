import os
import yaml  #pip install pyyaml
import shutil
import time
import argparse
import torch.nn as nn
import logging as log

import utils.log as logger
from pathlib import Path
from criteria import *
from dataloader import BatchDataLoader, SpeechMixDataset
from utils.Checkpoint import Checkpoint
from networks.FusionNet import FusionNet
from utils.progressbar import progressbar as pb
from utils.util import makedirs, saveYAML


def validate(network, eval_loader, weight, *criterion):
    network.eval()
    with torch.no_grad():
        cnt = 0.
        accu_eval_loss = 0.0
        ebar = pb(0, len(eval_loader.get_dataloader()), 20)
        ebar.start()
        for j, batch_eval in enumerate(eval_loader.get_dataloader()):
            features, labels = batch_eval[0].cuda(), batch_eval[1].cuda()
            outputs = network(features)
            loss = 0.
            for idx, cri in enumerate(criterion):
                loss += cri(outputs, batch_eval) * weight[idx]
            eval_loss = loss.data.item()
            accu_eval_loss += eval_loss
            cnt += 1.
            ebar.update_progress(j, 'CV   ', 'loss:{:.5f}/{:.5f}'.format(eval_loss, accu_eval_loss / cnt))

        avg_eval_loss = accu_eval_loss / cnt
    print()
    network.train()
    return avg_eval_loss


if __name__ == '__main__':

    """
    environment part
    """
    # loading argument
    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--model_name", help="trained model name, retrain if no input", default='none')
    parser.add_argument("-y", "--yaml_name", help="config file name")
    args = parser.parse_args()

    # loading config
    _abspath = Path(os.path.abspath(__file__)).parent
    _project = _abspath.stem
    _yaml_path = os.path.join(_abspath, 'configs/' + args.yaml_name)
    try:
        with open(_yaml_path, 'r') as f_yaml:
            config = yaml.load(f_yaml, Loader=yaml.FullLoader)
    except:
        raise ValueError('No config file found at "%s"' % _yaml_path)

    # make output dirs
    _outpath = config['OUTPUT_DIR'] + _project + config['WORKSPACE']
    _modeldir = _outpath + '/checkpoints/'
    _datadir = _outpath + '/estimations/'
    _logdir = _outpath + '/log/'
    makedirs([_modeldir, _datadir, _logdir])
    saveYAML(config, _outpath + '/' + args.yaml_name)

    logger.log(_logdir)
    """
    network part
    """
    # dataset
    tr_mix_dataset = SpeechMixDataset(config, mode='train')
    tr_batch_dataloader = BatchDataLoader(tr_mix_dataset, config['BATCH_SIZE'], is_shuffle=True,
                                          workers_num=config['NUM_WORK'])
    if config['USE_CV']:
        cv_mix_dataset = SpeechMixDataset(config, mode='validate')
        cv_batch_dataloader = BatchDataLoader(cv_mix_dataset, config['BATCH_SIZE'], is_shuffle=False,
                                              workers_num=config['NUM_WORK'])

    # device setting
    os.environ["CUDA_VISIBLE_DEVICES"] = config['CUDA_ID']

    # set model and optimizer
    network = FusionNet()
    network = nn.DataParallel(network)
    network.cuda()
    parameters = sum(p.numel() for p in network.parameters() if p.requires_grad)
    print("Trainable parameters : " + str(parameters))
    optimizer = torch.optim.Adam(network.parameters(), lr=config['LR'], amsgrad=True)
    lr_list = [0.0002] * 9 + [0.0001] * 18 + [0.00005] * 9 + [0.00001] * 24
    #  criteria,weight for each criterion
    criterion = mag_loss(config['WIN_LEN'], config['WIN_OFFSET'], loss_type='mse')
    weight = [1.]

    if args.model_name == 'none':
        log.info('#' * 12 + 'NO EXIST MODEL, TRAIN NEW MODEL ' + '#' * 12)
        best_loss = float('inf')
        start_epoch = 0
    else:
        checkpoint = Checkpoint()
        checkpoint.load(args.model_name)
        start_epoch = checkpoint.start_epoch
        best_loss = checkpoint.best_loss
        network.load_state_dict(checkpoint.state_dict)
        optimizer.load_state_dict(checkpoint.optimizer)
        log.info('#' * 18 + 'Finish Resume Model ' + '#' * 18)

    """
    training part
    """
    log.info('#' * 20 + ' START TRAINING ' + '#' * 20)
    cnt = 0.  #
    for epoch in range(start_epoch, config['MAX_EPOCH']):
        # set learning rate for every epoch
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr_list[epoch]

        # initial param
        accu_train_loss = 0.0
        network.train()
        tbar = pb(0, len(tr_batch_dataloader.get_dataloader()), 20)
        tbar.start()

        for i, batch_info in enumerate(tr_batch_dataloader.get_dataloader()):
            features, labels = batch_info[0].cuda(), batch_info[1].cuda()

            # forward + backward + optimize
            optimizer.zero_grad()
            t_outp, f_outp = network(features)

            t_loss = criterion(t_outp, batch_info)
            f_loss = criterion(t_outp, batch_info)
            loss = 0.85 * t_loss + 0.15 * f_loss
            loss.backward()
            optimizer.step()

            # calculate losses
            running_loss = loss.data.item()
            accu_train_loss += running_loss

            # display param
            cnt += 1
            # del loss, outputs, batch_info

            tbar.update_progress(i, 'Train', 'epoch:{}/{}, loss:{:.5f}/{:.5f}'.format(epoch + 1,
                                                                                      config['MAX_EPOCH'], running_loss,
                                                                                      accu_train_loss / cnt))
            if config['USE_CV'] and (i + 1) % config['EVAL_STEP'] == 0:
                print()
                avg_train_loss = accu_train_loss / cnt
                avg_eval_loss = validate(network, cv_batch_dataloader, weight, criterion)
                is_best = True if avg_eval_loss < best_loss else False
                best_loss = avg_eval_loss if is_best else best_loss
                log.info('Epoch [%d/%d], ( TrainLoss: %.4f | EvalLoss: %.4f )' % (
                    epoch + 1, config['MAX_EPOCH'], avg_train_loss, avg_eval_loss))

                checkpoint = Checkpoint(epoch + 1, avg_train_loss, best_loss, network.state_dict(),
                                        optimizer.state_dict())
                model_name = _modeldir + '{}-{}-val.ckpt'.format(epoch + 1, i + 1)
                best_model = _modeldir + 'best.ckpt'
                if is_best:
                    checkpoint.save(is_best, best_model)
                if not config['SAVE_BEST_ONLY']:
                    checkpoint.save(False, model_name)

                accu_train_loss = 0.0
                network.train()
                cnt = 0.

