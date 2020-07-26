
import time
import argparse
import os
import yaml
import shutil
import torch
from torch import nn
import torch.backends.cudnn as cudnn

from loader import get_hierdataset
from models import get_model
from optimizers import get_optimizer, step_scheduler
from metrics import averageMeter
from utils import get_logger, cvt2normal_state

from tensorboardX import SummaryWriter

def main():
    global lmbda, n_step, node_labels, n_nodes

    # setup seeds
    torch.manual_seed(cfg.get('seed', 1337))
    torch.cuda.manual_seed(cfg.get('seed', 1337))

    if not torch.cuda.is_available():
        raise SystemExit('GPU is needed')

    os.system('echo $CUDA_VISIBLE_DEVICES')

    # setup data loader
    data_loader = get_hierdataset(
        cfg['data'], cfg['training']['batch_size'], cfg['training']['n_workers'], ['train']
    )
    n_step = int(len(data_loader['train'].dataset) // float(cfg['training']['batch_size']))
    node_labels = data_loader['node_labels']
    nodes = data_loader['nodes']
    n_nodes = len(nodes)

    # setup Deep-RTC model (feature extractor + classifier)
    n_gpu = torch.cuda.device_count()
    model_fe = get_model(cfg['model']['fe']).cuda()
    model_fe = nn.DataParallel(model_fe, device_ids=range(n_gpu))

    model_cls = get_model(cfg['model']['cls'], nodes).cuda()
    model_cls = nn.DataParallel(model_cls, device_ids=range(n_gpu))

    model_pivot = get_model(cfg['model']['pivot']).cuda()
    model_pivot = nn.DataParallel(model_pivot, device_ids=range(n_gpu))

    # loss function
    criterion = nn.CrossEntropyLoss(reduction='none')
    lmbda = cfg['training']['lmbda']

    # setup optimizer
    opt_main_cls, opt_main_params = get_optimizer(cfg['training']['optimizer_main'])
    cnn_params = list(model_fe.parameters()) + list(model_cls.parameters())
    opt_main = opt_main_cls(cnn_params, **opt_main_params)
    logger.info('Using optimizer {}'.format(opt_main))

    # setup scheduler
    scheduler = step_scheduler(opt_main, **cfg['training']['scheduler'])

    cudnn.benchmark = True

    # load checkpoint
    start_ep = 0
    if cfg['training']['resume'].get('model', None):
        resume = cfg['training']['resume']
        if os.path.isfile(resume['model']):
            logger.info(
                "Loading model from checkpoint '{}'".format(resume['model'])
            )
            checkpoint = torch.load(resume['model'])
            model_fe.module.load_state_dict(cvt2normal_state(checkpoint['model_fe_state']))
            model_cls.module.load_state_dict(cvt2normal_state(checkpoint['model_cls_state']))
            if resume['param_only'] is False:
                start_ep = checkpoint['epoch']
                opt_main.load_state_dict(checkpoint['opt_main_state'])
                scheduler.load_state_dict(checkpoint['scheduler_state'])
            logger.info(
                "Loaded checkpoint '{}' (iter {})".format(
                    resume['model'], checkpoint['epoch']
                )
            )
        else:
            logger.info("No checkpoint found at '{}'".format(resume['model']))

    print('Start training from epoch {}'.format(start_ep))
    logger.info('Start training from epoch {}'.format(start_ep))

    for ep in range(start_ep, cfg['training']['epoch']):

        train(data_loader['train'], model_fe, model_cls, model_pivot, opt_main, ep, criterion)

        if (ep + 1) % cfg['training']['save_interval'] == 0:
            state = {
                'epoch': ep + 1,
                'model_fe_state': model_fe.state_dict(),
                'model_cls_state': model_cls.state_dict(),
                'opt_main_state': opt_main.state_dict(),
                'scheduler_state': scheduler.state_dict()
            }
            ckpt_path = os.path.join(writer.file_writer.get_logdir(), "ep-{ep}_model.pkl")
            save_path = ckpt_path.format(ep=ep+1)
            last_path = ckpt_path.format(ep=ep+1-cfg['training']['save_interval'])
            torch.save(state, save_path)
            if os.path.isfile(last_path):
                os.remove(last_path)
            print_str = '[Checkpoint]: {} saved'.format(save_path)
            print(print_str)
            logger.info(print_str)

        scheduler.step()


def train(data_loader, model_fe, model_cls, model_pivot, opt_main, epoch, criterion):

    # setup average meters
    batch_time = averageMeter()
    data_time = averageMeter()
    nlosses = averageMeter()
    stslosses = averageMeter()
    losses = averageMeter()
    acc = averageMeter()

    # setting training mode
    model_fe.train()
    model_cls.train()
    model_pivot.train()

    end = time.time()
    for (step, value) in enumerate(data_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        image = value[0].cuda()
        target = value[1].cuda(async=True)

        # forward
        _, imfeat = model_fe(x=image, feat=True)
        gate = model_pivot(torch.ones([image.size(0), n_nodes]))
        gate[:, 0] = 1
        output, nout, sfmx_base = model_cls(x=imfeat, gate=gate)

        # compute node-conditional consistency loss at each node for each sample
        nloss = []
        for idx in range(image.size(0)):
            for n_id, n_l in node_labels[value[1].numpy()[idx]]:
                nloss.append(criterion(nout[n_id][idx, :].view(1, -1), torch.tensor([n_l]).cuda()))

        nloss = torch.mean(torch.stack(nloss))
        nlosses.update(nloss.item(), image.size(0))

        # compute stochastic tree ssampling loss
        gt_z = torch.gather(output, 1, target.view(-1, 1))
        stsloss = torch.mean(-gt_z + torch.log(torch.clamp(sfmx_base.view(-1, 1), 1e-17, 1e17)))
        stslosses.update(stsloss.item(), image.size(0))

        loss = nloss + stsloss * lmbda
        losses.update(loss.item(), image.size(0))

        # measure accuracy
        max_z = torch.max(output, dim=1)[0]
        preds = torch.eq(output, max_z.view(-1, 1))
        iscorrect = torch.gather(preds, 1, target.view(-1, 1)).flatten().float()
        acc.update(torch.mean(iscorrect).item(), image.size(0))

        # back propagation
        opt_main.zero_grad()
        loss.backward()
        opt_main.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if (step + 1) % 10 == 0:
            curr_lr_main = opt_main.param_groups[0]['lr']
            print_str = 'Epoch [{0}/{1}]\t' \
                'Step: [{2}/{3}]\t' \
                'LR: [{4}]\t' \
                'Time {batch_time.avg:.3f}\t' \
                'Data {data_time.avg:.3f}\t' \
                'Loss {loss.avg:.4f}\t' \
                'Acc {acc.avg:.3f}'.format(
                    epoch + 1, cfg['training']['epoch'], step + 1, n_step, curr_lr_main, batch_time=batch_time,
                    data_time=data_time, loss=losses, acc=acc
                )

            print(print_str)
            logger.info(print_str)

    if (epoch + 1) % cfg['training']['print_interval'] == 0:
        curr_lr_main = opt_main.param_groups[0]['lr']
        print_str = 'Epoch: [{0}/{1}]\t' \
            'LR: [{2}]\t' \
            'Time {batch_time.avg:.3f}\t' \
            'Data {data_time.avg:.3f}\t' \
            'Loss {loss.avg:.4f}\t' \
            'Acc {acc.avg:.3f}'.format(
                epoch + 1, cfg['training']['epoch'], curr_lr_main, batch_time=batch_time,
                data_time=data_time, loss=losses, acc=acc
            )

        print(print_str)
        logger.info(print_str)
        writer.add_scalar('train/lr', curr_lr_main, epoch + 1)
        writer.add_scalar('train/nloss', nlosses.avg, epoch + 1)
        writer.add_scalar('train/stsloss', stslosses.avg, epoch + 1)
        writer.add_scalar('train/loss', losses.avg, epoch + 1)
        writer.add_scalar('train/acc', acc.avg, epoch + 1)


if __name__ == '__main__':
    global cfg, args, writer, logger

    parser = argparse.ArgumentParser(description='config')
    parser.add_argument(
        '--config',
        nargs='?',
        type=str,
        default='configs/inaturalist.yml',
        help='Configuration file to use',
    )

    args = parser.parse_args()

    with open(args.config) as fp:
        cfg = yaml.load(fp, Loader=yaml.SafeLoader)

    logdir = os.path.join('runs', os.path.basename(args.config)[:-4], cfg['exp'])
    writer = SummaryWriter(log_dir=logdir)

    print("RUNDIR: {}".format(logdir))
    shutil.copy(args.config, logdir)

    logger = get_logger(logdir)
    logger.info("Start logging")

    main()

