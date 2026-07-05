
import torch
import torch.nn as nn
import numpy as np
from medpy import metric
from scipy.ndimage import zoom, label as ndlabel, distance_transform_edt
import seaborn as sns
from PIL import Image 
import matplotlib.pyplot as plt
from segmentation_mask_overlay import overlay_masks
import matplotlib.colors as mcolors


import SimpleITK as sitk
import pandas as pd


from thop import profile
from thop import clever_format

def powerset(seq):
    """
    Returns all the subsets of this set. This is a generator.
    """
    if len(seq) <= 1:
        yield seq
        yield []
    else:
        for item in powerset(seq[1:]):
            yield [seq[0]]+item
            yield item

def clip_gradient(optimizer, grad_clip):
    """
    For calibrating misalignment gradient via cliping gradient technique
    :param optimizer:
    :param grad_clip:
    :return:
    """
    for group in optimizer.param_groups:
        for param in group['params']:
            if param.grad is not None:
                param.grad.data.clamp_(-grad_clip, grad_clip)


def adjust_lr(optimizer, init_lr, epoch, decay_rate=0.1, decay_epoch=30):
    decay = decay_rate ** (epoch // decay_epoch)
    for param_group in optimizer.param_groups:
        param_group['lr'] *= decay


class AvgMeter(object):
    def __init__(self, num=40):
        self.num = num
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0
        self.losses = []

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        self.losses.append(val)

    def show(self):
        return torch.mean(torch.stack(self.losses[np.maximum(len(self.losses)-self.num, 0):]))


def CalParams(model, input_tensor):
    """
    Usage:
        Calculate Params and FLOPs via [THOP](https://github.com/Lyken17/pytorch-OpCounter)
    Necessarity:
        from thop import profile
        from thop import clever_format
    :param model:
    :param input_tensor:
    :return:
    """
    flops, params = profile(model, inputs=(input_tensor,))
    flops, params = clever_format([flops, params], "%.3f")
    print('[Statistics Information]\nFLOPs: {}\nParams: {}'.format(flops, params))
    
def one_hot_encoder(input_tensor,dataset,n_classes = None):
    tensor_list = []
    if dataset == 'MMWHS':  
        dict = [0,205,420,500,550,600,820,850]
        for i in dict:
            temp_prob = input_tensor == i  
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()
    else:
        for i in range(n_classes):
            temp_prob = input_tensor == i  
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()    

class DiceLoss(nn.Module):
    def __init__(self, n_classes):
        super(DiceLoss, self).__init__()
        self.n_classes = n_classes

    def _one_hot_encoder(self, input_tensor):
        tensor_list = []
        for i in range(self.n_classes):
            temp_prob = input_tensor == i  # * torch.ones_like(input_tensor)
            tensor_list.append(temp_prob.unsqueeze(1))
        output_tensor = torch.cat(tensor_list, dim=1)
        return output_tensor.float()

    def _dice_loss(self, score, target):
        target = target.float()
        smooth = 1e-5
        intersect = torch.sum(score * target)
        y_sum = torch.sum(target * target)
        z_sum = torch.sum(score * score)
        loss = (2 * intersect + smooth) / (z_sum + y_sum + smooth)
        loss = 1 - loss
        return loss

    def forward(self, inputs, target, weight=None, softmax=False):
        if softmax:
            inputs = torch.softmax(inputs, dim=1)
        target = self._one_hot_encoder(target)
        if weight is None:
            weight = [1] * self.n_classes
        #print(inputs)
        assert inputs.size() == target.size(), 'predict {} & target {} shape do not match'.format(inputs.size(), target.size())
        class_wise_dice = []
        loss = 0.0
        for i in range(0, self.n_classes):
            dice = self._dice_loss(inputs[:, i], target[:, i])
            class_wise_dice.append(1.0 - dice.item())
            loss += dice * weight[i]
        return loss / self.n_classes


class BoundaryLoss(nn.Module):
    """Distance-map boundary loss (Kervadec et al., 2019).

    Precomputes a signed distance map per class from the GT mask (negative
    inside the object, positive outside, magnitude = distance to the
    boundary) and weights the predicted softmax probabilities by it. Unlike
    Dice/CE, a false positive far from the true boundary is penalized more
    than one right at the edge — it directly targets the "wrong region
    entirely" failure mode rather than just boundary sharpness.

    Has no lower bound of 0 like Dice/CE do, so it must be combined with a
    region loss at a small, ramped-up weight (start near 0, increase over
    the first N epochs) — using it alone or at full strength from epoch 0
    is known to destabilize training.
    """
    def __init__(self, n_classes):
        super().__init__()
        self.n_classes = n_classes

    @torch.no_grad()
    def _dist_maps(self, label):
        label_np = label.detach().cpu().numpy().astype(np.int64)
        b_sz, h, w = label_np.shape
        maps = np.zeros((b_sz, self.n_classes, h, w), dtype=np.float32)
        for b in range(b_sz):
            for c in range(self.n_classes):
                posmask = (label_np[b] == c)
                if posmask.any() and not posmask.all():
                    maps[b, c] = distance_transform_edt(~posmask) - distance_transform_edt(posmask)
                elif posmask.all():
                    maps[b, c] = -distance_transform_edt(posmask)
                # class absent everywhere in this slice -> leave at 0, no penalty either way
        return torch.from_numpy(maps)

    def forward(self, logits, label):
        probs = torch.softmax(logits, dim=1)
        dist_maps = self._dist_maps(label.long()).to(logits.device)
        return (probs * dist_maps).mean()


def predict_tta(net, t):
    """Flip test-time augmentation: average softmax over identity + H-flip + V-flip.
    t: (1, 1, H, W) input tensor. Returns (1, C, H, W) averaged probabilities."""
    probs = []
    for dims in ([], [2], [3]):
        inp = torch.flip(t, dims) if dims else t
        P = net(inp)
        p = torch.softmax(sum(P), dim=1)
        if dims:
            p = torch.flip(p, dims)
        probs.append(p)
    return torch.stack(probs).mean(0)


def keep_largest_component(volume, classes):
    """Keep only the largest connected component per foreground class over the
    whole 3D volume, dropping spurious islands the network predicts elsewhere.
    Test-time only post-processing — no retraining needed."""
    cleaned = np.zeros_like(volume)
    for c in range(1, classes):
        mask = (volume == c)
        if not mask.any():
            continue
        labeled, n = ndlabel(mask)
        if n <= 1:
            cleaned[mask] = c
            continue
        sizes = np.bincount(labeled.ravel())
        sizes[0] = 0
        cleaned[labeled == sizes.argmax()] = c
    return cleaned


def calculate_metric_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        hd95 = metric.binary.hd95(pred, gt)
        jaccard = metric.binary.jc(pred, gt)
        asd = metric.binary.assd(pred, gt)
        return dice, hd95, jaccard, asd
    elif pred.sum() > 0 and gt.sum()==0:
        return 1, 0, 1, 0
    else:
        return 0, 0, 0, 0

def calculate_dice_percase(pred, gt):
    pred[pred > 0] = 1
    gt[gt > 0] = 1
    if pred.sum() > 0 and gt.sum()>0:
        dice = metric.binary.dc(pred, gt)
        return dice
    elif pred.sum() > 0 and gt.sum()==0:
        return 1
    else:
        return 0


def test_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1, class_names=None):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if class_names==None:
        mask_labels = np.arange(1,classes)
    else:
        mask_labels = class_names
    cmaps = mcolors.CSS4_COLORS

    my_colors=['red','darkorange','yellow','forestgreen','blue','purple','magenta','cyan','deeppink', 'chocolate', 'olive','deepskyblue','darkviolet']
    cmap = {k: cmaps[k] for k in sorted(cmaps.keys()) if k in my_colors[:classes-1]}

    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                P = net(input)
                #print(len(P))
                outputs = 0.0
                for idx in range(len(P)):
                    outputs += P[idx]
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out             
                
                lbl = label[ind, :, :]
                masks = []
                for i in range(1, classes):
                    masks.append(lbl==i)
                preds_o = []
                for i in range(1, classes):
                    preds_o.append(pred==i)
                prediction[ind] = pred
    else:
        x, y = image.shape[0], image.shape[1]
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            P = net(input)
            outputs = 0.0
            for idx in range(len(P)):
                outputs += P[idx]
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
            if x != patch_size[0] or y != patch_size[1]:
                prediction = zoom(prediction, (x / patch_size[0], y / patch_size[1]), order=0)
    prediction = keep_largest_component(prediction, classes)
    metric_list = []

    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list


def test_single_volume_Lung(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1,
                       class_names=None):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if class_names == None:
        mask_labels = np.arange(1, classes)
    else:
        mask_labels = class_names
    cmaps = mcolors.CSS4_COLORS

    my_colors = ['red', 'darkorange', 'yellow', 'forestgreen', 'blue', 'purple', 'magenta', 'cyan', 'deeppink',
                 'chocolate', 'olive', 'deepskyblue', 'darkviolet']
    cmap = {k: cmaps[k] for k in sorted(cmaps.keys()) if k in my_colors[:classes - 1]}

    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                P = net(input)
                # print(len(P))
                outputs = 0.0
                for idx in range(len(P)):
                    outputs += P[idx]
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out

                lbl = label[ind, :, :]
                masks = []
                for i in range(1, classes):
                    masks.append(lbl == i)
                preds_o = []
                for i in range(1, classes):
                    preds_o.append(pred == i)
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            P = net(input)
            outputs = 0.0
            for idx in range(len(P)):
                outputs += P[idx]
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []

    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list

def test_single_volumePlyp(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1,
                       class_names=None):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.cpu().detach().numpy()
    input = torch.from_numpy(image).unsqueeze(0).float().cuda()
    if class_names == None:
        mask_labels = np.arange(1, classes)
    else:
        mask_labels = class_names
    cmaps = mcolors.CSS4_COLORS

    my_colors = ['red', 'darkorange', 'yellow', 'forestgreen', 'blue', 'purple', 'magenta', 'cyan', 'deeppink',
                 'chocolate', 'olive', 'deepskyblue', 'darkviolet']
    cmap = {k: cmaps[k] for k in sorted(cmaps.keys()) if k in my_colors[:classes - 1]}
    input = torch.from_numpy(image).unsqueeze(0).float().cuda()
    net.eval()
    with torch.no_grad():
        P = net(input)
        outputs = 0.0
        for idx in range(len(P)):
            outputs += P[idx]
        out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
        prediction = out.cpu().detach().numpy()
    metric_list = []
    print(prediction.shape, label.shape)
    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))
    return metric_list
def test_single_volume1(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1,
                       class_names=None):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if class_names == None:
        mask_labels = np.arange(1, classes)
    else:
        mask_labels = class_names
    cmaps = mcolors.CSS4_COLORS

    my_colors = ['red', 'darkorange', 'yellow', 'forestgreen', 'blue', 'purple', 'magenta', 'cyan', 'deeppink',
                 'chocolate', 'olive', 'deepskyblue', 'darkviolet']
    cmap = {k: cmaps[k] for k in sorted(cmaps.keys()) if k in my_colors[:classes - 1]}

    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                P = net(input)
                # print(len(P))
                outputs = 0.0
                for idx in range(len(P)):
                    outputs += P[idx]
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out

                lbl = label[ind, :, :]
                print(lbl)
                masks = []
                for i in range(1, classes):
                    masks.append(lbl == i)
                preds_o = []
                for i in range(1, classes):
                    preds_o.append(pred == i)

                # saving the groundtruth lables and output prediction maps for each frame
                fig_gt = overlay_masks(image[ind, :, :], masks, labels=mask_labels, colors=cmap, mask_alpha=0.5)
                fig_pred = overlay_masks(image[ind, :, :], preds_o, labels=mask_labels, colors=cmap, mask_alpha=0.5)
                # Do with that image whatever you want to do.
                fig_gt.savefig(test_save_path + '/' + case + '_' + str(ind) + '_gt.png', bbox_inches="tight", dpi=300)
                fig_pred.savefig(test_save_path + '/' + case + '_' + str(ind) + '_pred.png', bbox_inches="tight",
                                 dpi=300)
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            P = net(input)
            outputs = 0.0
            for idx in range(len(P)):
                outputs += P[idx]
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []

    for i in range(1, classes):
        metric_list.append(calculate_metric_percase(prediction == i, label == i))

    if test_save_path is not None:
        img_itk = sitk.GetImageFromArray(image.astype(np.float32))
        prd_itk = sitk.GetImageFromArray(prediction.astype(np.float32))
        lab_itk = sitk.GetImageFromArray(label.astype(np.float32))
        img_itk.SetSpacing((1, 1, z_spacing))
        prd_itk.SetSpacing((1, 1, z_spacing))
        lab_itk.SetSpacing((1, 1, z_spacing))
        sitk.WriteImage(prd_itk, test_save_path + '/' + case + "_pred.nii.gz")
        sitk.WriteImage(img_itk, test_save_path + '/' + case + "_img.nii.gz")
        sitk.WriteImage(lab_itk, test_save_path + '/' + case + "_gt.nii.gz")
    return metric_list


def val_single_volume(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()

    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
            
                P = net(input)
                #print(len(P))

                outputs = 0.0
                for idx in range(len(P)):
                   outputs += P[idx]
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():

            P = net(input)

            outputs = 0.0
            for idx in range(len(P)):
               outputs += P[idx]
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_dice_percase(prediction == i, label == i))
    return metric_list

def val_single_volume_1out(image, label, net, classes, patch_size=[256, 256], test_save_path=None, case=None, z_spacing=1):
    image, label = image.squeeze(0).cpu().detach().numpy(), label.squeeze(0).cpu().detach().numpy()
    if len(image.shape) == 3:
        prediction = np.zeros_like(label)
        for ind in range(image.shape[0]):
            slice = image[ind, :, :]
            x, y = slice.shape[0], slice.shape[1]
            if x != patch_size[0] or y != patch_size[1]:
                slice = zoom(slice, (patch_size[0] / x, patch_size[1] / y), order=3)  # previous using 0
            input = torch.from_numpy(slice).unsqueeze(0).unsqueeze(0).float().cuda()
            net.eval()
            with torch.no_grad():
                p1 = net(input)
                outputs = p1
                out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
                out = out.cpu().detach().numpy()
                if x != patch_size[0] or y != patch_size[1]:
                    pred = zoom(out, (x / patch_size[0], y / patch_size[1]), order=0)
                else:
                    pred = out
                prediction[ind] = pred
    else:
        input = torch.from_numpy(image).unsqueeze(
            0).unsqueeze(0).float().cuda()
        net.eval()
        with torch.no_grad():
            p1 = net(input)
            outputs = p1
            out = torch.argmax(torch.softmax(outputs, dim=1), dim=1).squeeze(0)
            prediction = out.cpu().detach().numpy()
    metric_list = []
    for i in range(1, classes):
        metric_list.append(calculate_dice_percase(prediction == i, label == i))
    return metric_list        
if __name__=='__main__':
    image=torch.randn((1,3,256,256))
    gt=torch.randn(1, 255, 256)
    from lib.networks import MERIT_Parallel_Modified3

    net = MERIT_Parallel_Modified3().cuda()
    acc=test_single_volumePlyp(image, gt, net, 2, patch_size=[256, 256], test_save_path=None, case=None,z_spacing=1,class_names=None)
    print(acc)
