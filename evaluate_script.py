# define test set
# define weights
# load model
# iterate through test and predict
# collect GT and preds during inference
# calculate simple MAE for people count
# calculate mAP
import argparse
import os
import warnings
import numpy as np
import torch
from torch.utils.data import DataLoader
import torchvision.transforms as standard_transforms
import cv2
import util.misc as utils
from crowd_datasets.SHHA.loading_data import loading_data
from models import build_model
from scipy.optimize import linear_sum_assignment
from tqdm import tqdm

warnings.filterwarnings('ignore')


class DeNormalize(object):
    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        for t, m, s in zip(tensor, self.mean, self.std):
            t.mul_(s).add_(m)
        return tensor


def get_args_parser():
    parser = argparse.ArgumentParser('Set parameters for P2PNet evaluation', add_help=False)

    # * Backbone
    parser.add_argument('--backbone', default='vgg16_bn', type=str,
                        help="name of the convolutional backbone to use")

    parser.add_argument('--row', default=2, type=int,
                        help="row number of anchor points")
    parser.add_argument('--line', default=2, type=int,
                        help="line number of anchor points")

    parser.add_argument('--weight_path', default='./ckpt/latest.pth',  # './weights/SHTechA.pth',  #
                        help='path where the trained weights saved')

    return parser


def vis(samples, targets, pred, vis_dir, des=None):
    '''
    samples -> tensor: [batch, 3, H, W]
    targets -> list of dict: [{'points':[], 'image_id': str}]
    pred -> list: [num_preds, 2]
    '''
    gts = [t['point'].tolist() for t in targets]

    pil_to_tensor = standard_transforms.ToTensor()

    restore_transform = standard_transforms.Compose([
        DeNormalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        standard_transforms.ToPILImage()
    ])
    # draw one by one
    for idx in range(samples.shape[0]):
        sample = restore_transform(samples[idx])
        sample = pil_to_tensor(sample.convert('RGB')).numpy() * 255
        sample_gt = sample.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()
        sample_pred = sample.transpose([1, 2, 0])[:, :, ::-1].astype(np.uint8).copy()

        max_len = np.max(sample_gt.shape)

        size = 2
        # draw gt
        for t in gts[idx]:
            sample_gt = cv2.circle(sample_gt, (int(t[0]), int(t[1])), size, (0, 255, 0), -1)
        # draw predictions
        for p in pred[idx]:
            sample_pred = cv2.circle(sample_pred, (int(p[0]), int(p[1])), size, (0, 0, 255), -1)

        name = targets[idx]['image_id']
        # save the visualized images
        if des is not None:
            cv2.imwrite(os.path.join(vis_dir, '{}_{}_gt_{}_pred_{}_gt.jpg'.format(int(name),
                                                des, len(gts[idx]), len(pred[idx]))), sample_gt)
            cv2.imwrite(os.path.join(vis_dir, '{}_{}_gt_{}_pred_{}_pred.jpg'.format(int(name),
                                                des, len(gts[idx]), len(pred[idx]))), sample_pred)
        else:
            cv2.imwrite(
                os.path.join(vis_dir, '{}_gt_{}_pred_{}_gt.jpg'.format(int(name), len(gts[idx]), len(pred[idx]))),
                sample_gt)
            cv2.imwrite(
                os.path.join(vis_dir, '{}_gt_{}_pred_{}_pred.jpg'.format(int(name), len(gts[idx]), len(pred[idx]))),
                sample_pred)


def calculate_map_metrics(predictions, targets, threshold=50):
    target_points = torch.cat([v["point"].cpu() for v in targets])
    matrix = torch.cdist(predictions, target_points, p=2)
    row_ind, col_ind = linear_sum_assignment(matrix)
    tp = fp = 0
    for row, col in zip(row_ind, col_ind):
        if matrix[row][col] <= threshold:
            tp += 1
        else:
            fp += 1
    fn = max(0, len(targets) - len(predictions))
    return tp, fp, fn


@torch.no_grad()
def evaluate_crowd_no_overlap(model, data_loader, device, vis_dir=None):
    model.eval()
    metric_logger = utils.MetricLogger(delimiter="  ")
    metric_logger.add_meter('class_error', utils.SmoothedValue(window_size=1, fmt='{value:.2f}'))
    # run inference on all images to calc MAE
    maes = []
    mses = []
    tps = []
    fps = []
    for samples, targets in tqdm(data_loader):  # assume that batch size == 1
        samples = samples.to(device)
        outputs = model(samples)
        outputs_scores = torch.nn.functional.softmax(outputs['pred_logits'], -1)[:, :, 1][0]
        outputs_points = outputs['pred_points'][0]

        gt_cnt = targets[0]['point'].shape[0]
        # 0.5 is used by default
        threshold = 0.5

        points = outputs_points[outputs_scores > threshold].detach().cpu().numpy().tolist()
        predict_cnt = int((outputs_scores > threshold).sum())
        # if specified, save the visualized images
        if vis_dir is not None:
            vis(samples, targets, [points], vis_dir)
        # accumulate MAE, MSE, mAP
        predictions = torch.Tensor(points).cpu()
        tp, fp, fn = calculate_map_metrics(predictions, targets)
        mae = abs(predict_cnt - gt_cnt)
        mse = (predict_cnt - gt_cnt) * (predict_cnt - gt_cnt)
        maes.append(float(mae))
        mses.append(float(mse))
        tps.append(tp)
        fps.append(fp)
        if len(tps) + len(fps) > 50:
            break
    # calc MAE, MSE
    mae = np.mean(maes)
    mse = np.sqrt(np.mean(mses))
    m_ap = sum(tps) / (sum(fps) + sum(tps))
    return mae, mse, m_ap


def main(args, debug=False):
    """
    This script evaluates p2p
    :param args:
    :param debug:
    :return:
    """
    # os.environ["CUDA_VISIBLE_DEVICES"] = '{}'.format(args.gpu_id)
    device = torch.device('cuda')
    data_root = './DATA_ROOT'
    output_dir = './EVAL_OUTPUT'

    # get the P2PNet
    model = build_model(args)
    # move to GPU
    model.to(device)
    # load trained model
    if args.weight_path is not None:
        checkpoint = torch.load(args.weight_path, map_location='cpu')
        model.load_state_dict(checkpoint['model'])
    # convert to eval mode
    model.eval()

    _, val_set = loading_data(data_root)
    data_loader_val = DataLoader(val_set, 1, drop_last=False, collate_fn=utils.collate_fn_crowd)
    result = evaluate_crowd_no_overlap(model, data_loader_val, device, vis_dir=output_dir)
    print(result)


if __name__ == '__main__':
    print(torch.cuda.is_available())
    parser = argparse.ArgumentParser('P2PNet training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main(args)
