import torch 
import torch.nn as nn
from torch.autograd import Variable
import numpy as np
import cv2 
import math
import os
from skimage.transform import resize
from model.EmptyLayer import EmptyLayer
from model.YOLOLayer import YOLOLayer


def parseModelConfig(model_cfg_path):
    assert os.path.exists(model_cfg_path), '{} not exists'.format(model_cfg_path)
    file = open(model_cfg_path, 'r')
    lines = file.read().split('\n')
    lines = [x for x in lines if x and not x.startswith('#')]
    lines = [x.rstrip().lstrip() for x in lines]
    blocks = list()
    for line in lines:
        if line.startswith('['):
            blocks.append(dict())
            blocks[-1]['type'] = line[1:-1].rstrip()
        else:
            key, value = line.split('=')
            blocks[-1][key.rstrip()] = value.lstrip()

    return blocks

def createModules(blocks):
    net_info = blocks[0]   
    module_list = nn.ModuleList()
    index = 0  
    prev_filters = 3
    output_filters = []
    
    for x in blocks:
        module = nn.Sequential()
        if (x["type"] == "net"):
            continue
        if (x["type"] == "convolutional"):
            activation = x["activation"]
            try:
                batch_normalize = int(x["batch_normalize"])
                bias = False
            except:
                batch_normalize = 0
                bias = True
            filters= int(x["filters"])
            padding = int(x["pad"])
            kernel_size = int(x["size"])
            stride = int(x["stride"])
            if padding:
                pad = (kernel_size - 1) // 2
            else:
                pad = 0

            conv = nn.Conv2d(prev_filters, filters, kernel_size, stride, pad, bias = bias)
            module.add_module("conv_{0}".format(index), conv)
            if batch_normalize:
                bn = nn.BatchNorm2d(filters)
                module.add_module("batch_norm_{0}".format(index), bn)

            if activation == "leaky":
                activn = nn.LeakyReLU(0.1, inplace = True)
                module.add_module("leaky_{0}".format(index), activn)

        elif (x["type"] == "upsample"):
            stride = int(x["stride"])
            upsample = nn.Upsample(scale_factor = 2, mode = "nearest")
            module.add_module("upsample_{}".format(index), upsample)

        elif (x["type"] == "route"):
            x["layers"] = x["layers"].split(',')
            start = int(x["layers"][0])
            try:
                end = int(x["layers"][1])
            except:
                end = 0
            if start > 0: 
                start = start - index
            
            if end > 0:
                end = end - index
            route = EmptyLayer()
            module.add_module("route_{0}".format(index), route)
            
            if end < 0:
                filters = output_filters[index + start] + output_filters[index + end]
            else:
                filters= output_filters[index + start]
                        
        elif x["type"] == "shortcut":
            from_ = int(x["from"])
            shortcut = EmptyLayer()
            module.add_module("shortcut_{}".format(index), shortcut)
            
        elif x["type"] == "yolo":
            mask = x["mask"].split(",")
            mask = [int(x) for x in mask]
            
            anchors = x["anchors"].split(",")
            anchors = [int(a) for a in anchors]
            anchors = [(anchors[i], anchors[i+1]) for i in range(0, len(anchors),2)]
            anchors = [anchors[i] for i in mask]
            
            detection = YOLOLayer(anchors)
            module.add_module("Detection_{}".format(index), detection)
        else:
            print("Something I dunno")
            assert False

        module_list.append(module)
        prev_filters = filters
        output_filters.append(filters)
        index += 1
        
    return (net_info, module_list)

def loadClass(namesfile):
    fp = open(namesfile, "r")
    names = fp.read().split("\n")[:-1]
    return names

def unique(tensor):
    tensor_np = tensor.cpu().numpy()
    unique_np = np.unique(tensor_np)
    unique_tensor = torch.from_numpy(unique_np)
    
    tensor_res = tensor.new(unique_tensor.shape)
    tensor_res.copy_(unique_tensor)
    return tensor_res

def nonMaxSuppression(prediction, confidence, num_classes, nms = True, nms_conf = 0.4):
    conf_mask = (prediction[:,:,4] > confidence).float().unsqueeze(2)
    prediction = prediction*conf_mask
    try:
        ind_nz = torch.nonzero(prediction[:,:,4]).transpose(0,1).contiguous()
    except:
        return 0
    
    box_a = prediction.new(prediction.shape)
    box_a[:,:,0] = (prediction[:,:,0] - prediction[:,:,2]/2)
    box_a[:,:,1] = (prediction[:,:,1] - prediction[:,:,3]/2)
    box_a[:,:,2] = (prediction[:,:,0] + prediction[:,:,2]/2) 
    box_a[:,:,3] = (prediction[:,:,1] + prediction[:,:,3]/2)
    prediction[:,:,:4] = box_a[:,:,:4]
    
    batch_size = prediction.size(0)
    
    output = prediction.new(1, prediction.size(2) + 1)
    write = False

    for ind in range(batch_size):
        image_pred = prediction[ind]
        
        max_conf, max_conf_score = torch.max(image_pred[:,5:5+ num_classes], 1)
        max_conf = max_conf.float().unsqueeze(1)
        max_conf_score = max_conf_score.float().unsqueeze(1)
        seq = (image_pred[:,:5], max_conf, max_conf_score)
        image_pred = torch.cat(seq, 1)
        
        non_zero_ind =  (torch.nonzero(image_pred[:,4]))
        image_pred_ = image_pred[non_zero_ind.squeeze(),:].view(-1,7)
        
        try:
            img_classes = unique(image_pred_[:,-1])
        except:
             continue
        for cls in img_classes:
            cls_mask = image_pred_*(image_pred_[:,-1] == cls).float().unsqueeze(1)
            class_mask_ind = torch.nonzero(cls_mask[:,-2]).squeeze()
            
            image_pred_class = image_pred_[class_mask_ind].view(-1,7)

            conf_sort_index = torch.sort(image_pred_class[:,4], descending = True )[1]
            image_pred_class = image_pred_class[conf_sort_index]
            idx = image_pred_class.size(0)
            
            if nms:
                for i in range(idx):
                    try:
                        ious = bboxIOU(image_pred_class[i].unsqueeze(0), image_pred_class[i+1:],True)
                    except ValueError:
                        break
        
                    except IndexError:
                        break
                    
                    iou_mask = (ious < nms_conf).float().unsqueeze(1)
                    image_pred_class[i+1:] *= iou_mask       
                    
                    non_zero_ind = torch.nonzero(image_pred_class[:,4]).squeeze()
                    image_pred_class = image_pred_class[non_zero_ind].view(-1,7)
            
            batch_ind = image_pred_class.new(image_pred_class.size(0), 1).fill_(ind)
            seq = batch_ind, image_pred_class
            if not write:
                output = torch.cat(seq,1)
                write = True
            else:
                out = torch.cat(seq,1)
                output = torch.cat((output,out))
    
    return output

def parseDataConfig(data_cfg_path):
    assert os.path.exists(data_cfg_path), '{} not exists'.format(data_cfg_path)
    options = dict()
    lines = open(data_cfg_path,'r').readlines()
    for line in lines:
        line = line.strip()
        if line == '' or line.startswith('#'):
            continue
        key, value = line.split('=')
        options[key.strip()] = value.strip()
    return options

def buildTargets(pred_boxes,pred_conf,pred_cls,targets,anchors,num_anchors,num_classes,grid_size,ignore_thres,img_size):
    batch_size = targets.size(0)
    mask = torch.zeros(batch_size,num_anchors,grid_size,grid_size)
    conf_mask = torch.ones(batch_size,num_anchors,grid_size,grid_size)
    tx = torch.zeros(batch_size,num_anchors,grid_size,grid_size)
    ty = torch.zeros(batch_size,num_anchors,grid_size,grid_size)
    tw = torch.zeros(batch_size,num_anchors,grid_size,grid_size)
    th = torch.zeros(batch_size,num_anchors,grid_size,grid_size)
    tconf = torch.ByteTensor(batch_size,num_anchors,grid_size,grid_size).fill_(0)
    tcls = torch.ByteTensor(batch_size,num_anchors,grid_size,grid_size,num_classes).fill_(0)

    num_ground_truth = 0
    num_correct = 0
    for batch_idx in range(batch_size):
        for target_idx in range(targets.shape[1]):
            # there is no target, continue
            if targets[batch_idx, target_idx].sum() == 0:
                continue
            num_ground_truth += 1

            # convert to position relative to bounding box
            gx = targets[batch_idx,target_idx, 1] * grid_size
            gy = targets[batch_idx,target_idx, 2] * grid_size
            gw = targets[batch_idx,target_idx, 3] * grid_size
            gh = targets[batch_idx,target_idx, 4] * grid_size

            # get grid box indices
            gi = int(gx)
            gj = int(gy)

            '''
            get the anchor box that has the highest iou with [gw, gh]
            '''
            # shape of the gt box
            gt_box = torch.FloatTensor(np.array([0, 0, gw, gh])).unsqueeze(0)
            # shape of anchor box
            anchor_shapes = torch.FloatTensor(np.concatenate((np.zeros((len(anchors), 2)), np.array(anchors)), 1))
            # get iou
            anchor_iou = bboxIOU(gt_box, anchor_shapes, True)
            # ingore iou that is larger than some threshold
            conf_mask[batch_idx, anchor_iou > ignore_thres, gj, gi] = 0
            # best matching anchor box
            best = np.argmax(anchor_iou)
            '''
            calculate the best iou between target and best pred box
            '''
            # ground truth box
            gt_box = torch.FloatTensor(np.array([gx, gy, gw, gh])).unsqueeze(0)
            # best pred box
            pred_box = pred_boxes[batch_idx, best, gj, gi].type(torch.FloatTensor).unsqueeze(0)
            mask[batch_idx, best, gj, gi] = 1
            conf_mask[batch_idx, best, gj, gi] = 1

            '''
            get target box dimension that is relative to gird rather than the entire
            input image as which may come in different dimensions. Grid size is fixed.
            So we predict position that is relative to gird.
            '''
            tx[batch_idx, best, gj, gi] = gx - gi
            ty[batch_idx, best, gj, gi] = gy - gj
            tw[batch_idx, best, gj, gi] = math.log(gw / anchors[best][0] + 1e-16)
            th[batch_idx, best, gj, gi] = math.log(gh / anchors[best][1] + 1e-16)

            target_label = int(targets[batch_idx, target_idx, 0])
            tcls[batch_idx, best, gj, gi, target_label] = 1
            tconf[batch_idx, best, gj, gi] = 1

            # calculate iou
            iou = bboxIOU(gt_box, pred_box, False)
            pred_label = torch.argmax(pred_cls[batch_idx, best, gj, gi])
            score = pred_conf[batch_idx, best, gj, gi]
            if iou > 0.5 and pred_label == target_label and score > 0.5:
                num_correct += 1

    return num_ground_truth, num_correct, mask, conf_mask, tx, ty, tw, th, tconf, tcls

def bboxIOU(box1, box2, x1y1x2y2):
    if x1y1x2y2:
        b1_x1, b1_y1, b1_x2, b1_y2 = box1[:, 0], box1[:, 1], box1[:, 2], box1[:, 3]
        b2_x1, b2_y1, b2_x2, b2_y2 = box2[:, 0], box2[:, 1], box2[:, 2], box2[:, 3]
    else:
        # convert center to exact coordinates
        b1_x1, b1_x2 = box1[:, 0] - box1[:, 2] / 2, box1[:, 0] + box1[:, 2] / 2
        b1_y1, b1_y2 = box1[:, 1] - box1[:, 3] / 2, box1[:, 1] + box1[:, 3] / 2
        b2_x1, b2_x2 = box2[:, 0] - box2[:, 2] / 2, box2[:, 0] + box2[:, 2] / 2
        b2_y1, b2_y2 = box2[:, 1] - box2[:, 3] / 2, box2[:, 1] + box2[:, 3] / 2

    intersect_x1 = torch.max(b1_x1, b2_x1)
    intersect_y1 = torch.max(b1_y1, b2_y1)
    intersect_x2 = torch.min(b1_x2, b2_x2)
    intersect_y2 = torch.min(b1_y2, b2_y2)

    intersect_area = (intersect_x2 - intersect_x1 + 1) * (intersect_y2 - intersect_y1 + 1)

    # union area
    b1_area = (b1_x2 - b1_x1 + 1) * (b1_y2 - b1_y1 + 1)
    b2_area = (b2_x2 - b2_x1 + 1) * (b2_y2 - b2_y1 + 1)

    iou = intersect_area/(b1_area+b2_area-intersect_area+1e-16)

    return iou

def prepImage(img, img_size):
    orig_img = img
    h, w, _ = img.shape
    dim_diff = np.abs(h-w)
    pad1, pad2 = dim_diff // 2, dim_diff - dim_diff // 2
    pad = ((pad1,pad2),(0,0),(0,0)) if h <= w else ((0,0),(pad1,pad2),(0,0))
    input_img = np.pad(img, pad, 'constant', constant_values=128)/255.0
    input_img = resize(input_img, (img_size, img_size, 3), mode='reflect',anti_aliasing=True)
    input_img = np.transpose(input_img, (2,0,1))
    input_img = torch.from_numpy(input_img).float().unsqueeze(0)

    return orig_img, input_img

def bboxIOUNumpy(box1, box2):
    area = (box2[:, 2] - box2[:, 0]) * (box2[:, 3] - box2[:, 1])

    iw = np.minimum(np.expand_dims(box1[:, 2], axis=1), box2[:, 2]) - np.maximum(
        np.expand_dims(box1[:, 0], 1), box2[:, 0])
    ih = np.minimum(np.expand_dims(box1[:, 3], axis=1), box2[:, 3]) - np.maximum(
        np.expand_dims(box1[:, 1], 1), box2[:, 1])
    iw = np.maximum(iw, 0)
    ih = np.maximum(ih, 0)

    ua = np.expand_dims((box1[:, 2] - box1[:, 0]) * (box1[:, 3] - box1[:, 1]), axis=1) + area - iw * ih
    ua = np.maximum(ua, np.finfo(float).eps)

    intersection = iw * ih

    return intersection / ua

def computeAP(recall, precision):
    # first append sentinel values at the end
    mrec = np.concatenate(([0.0], recall, [1.0]))
    mpre = np.concatenate(([0.0], precision, [0.0]))
    print('mrec',mrec)
    print('mpre',mpre)
    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
        mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])

    print('mpre',mpre)
    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]
    print(mrec[1:], mrec[:-1], i)

    # and sum (\Delta recall) * prec
    ap = np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])
    return ap