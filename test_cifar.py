
import os
import argparse
import numpy as np
import yaml
import logging
import torch

from loader import get_hierdataset_cifar
from models import get_model
from metrics import averageMeter
from utils import cvt2normal_state

def main():
    global num_class

    if not torch.cuda.is_available():
        raise SystemExit('GPU is needed')

    os.system('echo $CUDA_VISIBLE_DEVICES')

    # setup data loader
    data_loader = get_hierdataset_cifar(
        cfg['data'], args.b, 1, ['valid', 'test']
    )
    nodes = data_loader['nodes']
    num_class = len(nodes[0].codeword[0])

    # setup Deep-RTC model (feature extractor + classifier)
    model_fe = get_model(cfg['model']['fe']).cuda()
    model_cls = get_model(cfg['model']['cls'], nodes).cuda()

    # load checkpoint
    resume = os.path.join(args.checkpoint, 'ep-{ep}_model.pkl'.format(ep=cfg['training']['epoch']))

    if os.path.isfile(resume):
        checkpoint = torch.load(resume)
        model_fe.load_state_dict(cvt2normal_state(checkpoint['model_fe_state']))
        model_cls.load_state_dict(cvt2normal_state(checkpoint['model_cls_state']))
        print_str = "Loaded checkpoint '{}' (epoch {})".format(
            resume, checkpoint['epoch']
        )
        print(print_str)
        logger.info(print_str)
    else:
        print("No checkpoint found at '{}'".format(resume))
        logger.info("No checkpoint found at '{}'".format(resume))

    with torch.no_grad():
        # select the threshold on validation set
        CPB = []
        thres = np.linspace(0, 0.9, 10)
        for th in thres:
            _, cpb = test(data_loader['valid'], model_fe, model_cls, th)
            CPB.append(cpb)

        th_idx = np.argmax(CPB)

        # apply the threshold on testing set
        lacc, _ = test(data_loader['test'], model_fe, model_cls, 0)
        hacc, cpb = test(data_loader['test'], model_fe, model_cls, thres[th_idx])

        print_str = '[thres: {thres:.4f}]\t' \
            'leaf acc {lacc:.4f}\t' \
            'hier acc {hacc:.4f}\t' \
            'CPB {cpb:.4f}'.format(
                thres=thres[th_idx], lacc=lacc, hacc=hacc, cpb=cpb
        )

        print(print_str)
        logger.info(print_str)


def test(data_loader, model_fe, model_cls, thres):

    # setup average meter
    ACC = averageMeter()
    CPB = averageMeter()

    # set evaluation mode
    model_fe.eval()
    model_cls.eval()

    for (step, value) in enumerate(data_loader):

        image = value[0].cuda()
        target = value[1].cuda(async=True)

        # forward
        _, imfeat = model_fe(x=image, feat=True)
        output, nout = model_cls(x=imfeat, gate=None, pred=True, thres=thres)

        # get predictions
        max_z = torch.max(output, dim=1)[0]
        preds = torch.eq(output, max_z.view(-1, 1))

        # get boolean of correct prediction (correct: 1, incorrect: 0) and measure accuracy
        iscorrect = torch.gather(preds, 1, target.view(-1, 1)).flatten().float().cpu().data.numpy()
        ACC.update(np.mean(iscorrect), image.size(0))

        # compute PB & CPB
        Y_v = torch.sum(preds.float(), dim=1).data.cpu().numpy()
        Y_v[Y_v == 1] = 0
        pb = (num_class - Y_v) / num_class
        cpb = pb * iscorrect
        CPB.update(np.mean(cpb), image.size(0))

    return ACC.avg, CPB.avg


if __name__ == '__main__':
    global cfg, args, logger

    parser = argparse.ArgumentParser(description='config')
    parser.add_argument(
        '--checkpoint',
        type=str,
        default='runs/inaturalist/deep-rtc',
        help='checkpoint to evaluate',
    )
    parser.add_argument(
        '--b',
        type=int,
        default=4,
        help='batch size',
    )
    args = parser.parse_args()

    basename = args.checkpoint.split('/')[1]
    config = os.path.join(args.checkpoint, '{}.yml'.format(basename))
    with open(config) as fp:
        cfg = yaml.load(fp, Loader=yaml.SafeLoader)

    logger = logging.getLogger('mylogger')
    logfile = os.path.join(args.checkpoint, 'test_result.log')
    hdlr = logging.FileHandler(logfile)
    logger.addHandler(hdlr)
    logger.setLevel(logging.INFO)

    print('evaluating {}'.format(args.checkpoint))
    logger.info('evaluating {}'.format(args.checkpoint))

    main()

