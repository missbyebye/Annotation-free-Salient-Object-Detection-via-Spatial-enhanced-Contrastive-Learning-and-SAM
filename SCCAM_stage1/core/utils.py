import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import random

def cross_entropy2d(logit, target, ignore_index=255, weight=None, size_average=True, batch_average=True):
	n, c, h, w = logit.size()
	# logit = logit.permute(0, 2, 3, 1)
	target = target.squeeze(1)
	if weight is None:
		criterion = nn.CrossEntropyLoss(weight=weight, ignore_index=ignore_index, size_average=False)
	else:
		criterion = nn.CrossEntropyLoss(weight=torch.from_numpy(np.array(weight)).float().cuda(), ignore_index=ignore_index, size_average=False)
	loss = criterion(logit, target.long())

	if size_average:
		loss /= (h * w)

	if batch_average:
		loss /= n

	return loss

def BCE_2d(logit, target, ignore_index=255, weight=None, size_average=True, batch_average=True):
	n, c, h, w = logit.size()
	weight = 2 * abs(target - 0.5) * 0.0 + 1.0

	if weight is None:
		criterion = nn.BCELoss(weight=weight, size_average=False)
	else:
		criterion = nn.BCELoss(weight=weight, size_average=False)
	loss = criterion(logit, target)

	if size_average:
		loss /= (h * w)
	
	if batch_average:
		loss /= n
	
	return loss

def map_spatial_loss(maps):
    loss_sum = 0
 
    H = maps.size()[2]
    W = maps.size()[3]
    line = torch.linspace(0, W - 1, steps=W)
    line2= line * line
    pos_x = line.unsqueeze(0).repeat(H, 1).cuda()
    pos_y = pos_x.T
    pos_x2= line2.unsqueeze(0).repeat(H, 1).cuda()
    pos_y2= pos_x2.T

    for op_idx in range(maps.shape[0]):
        map_fg = maps[op_idx][0]
        map_bg = 1 - map_fg

        pos_x_fg_mean = torch.sum(map_fg * pos_x, dim=(0, 1)) / max(torch.sum(map_fg), 1)
        pos_y_fg_mean = torch.sum(map_fg * pos_y, dim=(0, 1)) / max(torch.sum(map_fg), 1)
        pos_x_bg_mean = torch.sum(map_bg * pos_x, dim=(0, 1)) / max(torch.sum(map_bg), 1)
        pos_y_bg_mean = torch.sum(map_bg * pos_y, dim=(0, 1)) / max(torch.sum(map_bg), 1)

        pos_x_cov_fg = torch.sum(map_fg * pos_x2, dim=(0,1)) / max(torch.sum(map_fg), 1) - pos_x_fg_mean ** 2
        pos_y_cov_fg = torch.sum(map_fg * pos_y2, dim=(0,1)) / max(torch.sum(map_fg), 1) - pos_y_fg_mean ** 2
        pos_x_cov_bg = torch.sum(map_bg * pos_x2, dim=(0,1)) / max(torch.sum(map_bg), 1) - pos_x_bg_mean ** 2
        pos_y_cov_bg = torch.sum(map_bg * pos_y2, dim=(0,1)) / max(torch.sum(map_bg), 1) - pos_y_bg_mean ** 2

        #loss_sum = loss_sum + torch.clamp(torch.exp((pos_x_cov_fg - pos_x_cov_bg + pos_y_cov_fg - pos_y_cov_bg) / pos_x.shape[0]), min=0, max=1)
        loss_sum = loss_sum + (pos_x_cov_fg - pos_x_cov_bg + pos_y_cov_fg - pos_y_cov_bg) / pos_x.shape[0]
    return torch.clamp(loss_sum, min=0) / maps.shape[0]



def BBox_Get(input):
    #op_threshold = torch.mean(input)
    op_threshold = torch.max(input) * 0.5
    op_mask=input.ge(op_threshold)
    x1 = input.shape[1]
    x2 = 0
    y1 = input.shape[0]
    y2 = 0
    #if torch.nonzero(op_mask).shape[0] > 0
    if torch.sum(op_mask) > 0:
        #sum in x
        op_sum = torch.sum(op_mask, 1)
        op_nz = torch.nonzero(op_sum)
        y1 = op_nz[0][0]
        y2 = op_nz[-1][0] + 1
        #sum in y
        op_sum = torch.sum(op_mask, 0)
        op_nz = torch.nonzero(op_sum)
        x1 = op_nz[0][0]
        x2 = op_nz[-1][0] + 1
        
    return x1, y1, x2, y2

def BBox_Area(x1, y1, x2, y2):
    if(x1 < x2 and y1 < y2):
        return ((x2 - x1) * (y2 - y1)).item()
    else:
        return 0
        
def BBox_loss2_old(predictions, bboxs):
    assert predictions.shape[0] == bboxs.shape[0]
    
    loss_sum = 0
    for op_idx in range(predictions.shape[0]):
        prediction = predictions[op_idx][0]
        x1_pred, y1_pred, x2_pred, y2_pred = BBox_Get(prediction)
        bbox = bboxs[op_idx][0]
        x1_bbox, y1_bbox, x2_bbox, y2_bbox = BBox_Get(bbox)
        
        x1_union = max(x1_pred, x1_bbox)
        x2_union = min(x2_pred, x2_bbox)
        y1_union = max(y1_pred, y1_bbox)
        y2_union = min(y2_pred, y2_bbox)
        
        area_pred = BBox_Area(x1_pred, y1_pred, x2_pred, y2_pred)
        area_bbox = BBox_Area(x1_bbox, y1_bbox, x2_bbox, y2_bbox)
        area_union = BBox_Area(x1_union, y1_union, x2_union, y2_union)
        
        loss = 1 - area_union / (area_pred + area_bbox - area_union + 0.000001)
        loss_sum = loss_sum + loss
    return loss_sum / predictions.shape[0] 


def BBox_loss_simple(predictions, bboxs):
    assert predictions.shape[0] == bboxs.shape[0]

    loss_sum = 0
    for op_idx in range(predictions.shape[0]):
        prediction = predictions[op_idx][0]
        proj_x_pred, idx = torch.max(prediction, 1)
        proj_y_pred, idx = torch.max(prediction, 0)
        
        bbox = bboxs[op_idx][0] > 0
        proj_x_bbox, idx  = torch.max(bbox, 1)
        proj_y_bbox, idx  = torch.max(bbox, 0)

        smooth = 1e-5
        intersection = (proj_x_pred * proj_x_bbox).sum(0)
        unionset = proj_x_pred.sum(0) + proj_x_bbox.sum(0)
        loss_x = 1 - (2 * intersection + smooth) / (unionset + smooth)

        intersection = (proj_y_pred * proj_y_bbox).sum(0)
        unionset = proj_y_pred.sum(0) + proj_y_bbox.sum(0)
        loss_y = 1 - (2 * intersection + smooth) / (unionset + smooth)

        loss_sum = loss_sum + (loss_x + loss_y) / 2
    return loss_sum / predictions.shape[0]


def BBox_loss(predictions, bboxs):
    assert predictions.shape[0] == bboxs.shape[0]

    loss_sum = 0
    for op_idx in range(predictions.shape[0]):
        bbox_full = bboxs[op_idx][0]
        prediction_full = predictions[op_idx][0]

        boxVal = 0.9
        iou_full = 0
        weight_full = 0
        intersection_full = 0
        unionsection_full = 0
        for scale in range(4):  #support max 4 objects

            mask = ((bbox_full >= boxVal) & (bbox_full < 2 * boxVal)).type(torch.uint8)
            bbox = torch.mul(bbox_full, mask)
            prediction = torch.mul(prediction_full, mask)

            if bbox.max() <= 0:
                break

            proj_x_pred, idx = torch.max(prediction, 1)
            proj_y_pred, idx = torch.max(prediction, 0)


            proj_x_bbox, idx  = torch.max(bbox, 1)
            proj_y_bbox, idx  = torch.max(bbox, 0)

            #if bbox.max() > 0:
            proj_x_bbox /= bbox.max()
            proj_y_bbox /= bbox.max()

            """
            smooth = 1e-5
            intersection = (proj_x_pred * proj_x_bbox).sum(0)
            unionset = proj_x_pred.sum(0) + proj_x_bbox.sum(0)
            iou_x = (intersection + smooth) / (unionset - intersection + smooth)

            intersection = (proj_y_pred * proj_y_bbox).sum(0)
            unionset = proj_y_pred.sum(0) + proj_y_bbox.sum(0)
            iou_y = (intersection + smooth) / (unionset - intersection + smooth)
       
            iou = iou_x * iou_y
            iou_full += iou * 1#torch.sum(mask)
            weight_full += 1#torch.sum(mask)
            """
            intersection = proj_x_pred.sum(0) * proj_y_pred.sum(0)
            unionsection = proj_x_bbox.sum(0) * proj_y_bbox.sum(0) #equal to mask.sum()
            iou_full += intersection / (unionsection + 1e-1)
            intersection_full += intersection
            boxVal /= 2

        mask_fg = (bbox_full >= boxVal).type(torch.uint8)
        mask_bg = (bbox_full < boxVal).type(torch.uint8)
        prediction_fg = torch.mul(prediction_full, mask_fg)
        prediction_bg = torch.mul(prediction_full, mask_bg)
        #pred_fg_ratio = prediction_fg.sum() / (prediction_full.sum() + 1e-1)
        #unionsection_full += prediction_full.sum() - prediction_fg.sum()
        proj_x_pred_fg, idx = torch.max(prediction_fg, 1)
        proj_y_pred_fg, idx = torch.max(prediction_fg, 0)
        proj_x_pred_bg, idx = torch.max(prediction_bg, 1)
        proj_y_pred_bg, idx = torch.max(prediction_bg, 0)

        iou_bg = intersection_full / (intersection_full + proj_x_pred_bg.sum(0) * proj_y_pred_bg.sum(0) + 1e-1)
        iou_all = iou_full / (scale + 1e-2) * iou_bg

        #loss_sum = loss_sum + (1 - (iou_full / (weight_full + 1e-1) *  pred_fg_ratio) )
        #loss_sum += 1 - intersection_full / (unionsection_full + 1e-1)
        loss_sum += 1 - iou_all
        #loss_sum = loss_sum + (1 - (iou_full / (weight_full + 1e-1) +  pred_fg_ratio) / 2) 
    return loss_sum / predictions.shape[0]


def BBox_loss3(predictions, bboxs):
    assert predictions.shape[0] == bboxs.shape[0]

    loss_sum = 0
    for op_idx in range(predictions.shape[0]):
        bbox_full = bboxs[op_idx][0]
        prediction_full = predictions[op_idx][0]

        boxVal = 0.9
        iou_full = 0
        weight_full = 0
        intersection_full = 0
        unionsection_full = 0
        for scale in range(4):  #support max 4 objects
            
            mask = ((bbox_full >= boxVal) & (bbox_full < 2 * boxVal)).type(torch.uint8)
            bbox = torch.mul(bbox_full, mask)
            prediction = torch.mul(prediction_full, mask)
            
            if bbox.max() <= 0:
                break

            proj_x_pred, idx = torch.max(prediction, 1)
            proj_y_pred, idx = torch.max(prediction, 0)


            proj_x_bbox, idx  = torch.max(bbox, 1)
            proj_y_bbox, idx  = torch.max(bbox, 0)

            #if bbox.max() > 0:
            proj_x_bbox /= bbox.max()
            proj_y_bbox /= bbox.max()

            """
            smooth = 1e-5
            intersection = (proj_x_pred * proj_x_bbox).sum(0)
            unionset = proj_x_pred.sum(0) + proj_x_bbox.sum(0)
            iou_x = (intersection + smooth) / (unionset - intersection + smooth)

            intersection = (proj_y_pred * proj_y_bbox).sum(0)
            unionset = proj_y_pred.sum(0) + proj_y_bbox.sum(0)
            iou_y = (intersection + smooth) / (unionset - intersection + smooth)
        
            iou = iou_x * iou_y
            iou_full += iou * 1#torch.sum(mask)
            """
            intersection_full += proj_x_pred.sum(0) * proj_y_pred.sum(0)
            unionsection_full += proj_x_bbox.sum(0) * proj_y_bbox.sum(0) #equal to mask.sum()
            weight_full += 1#torch.sum(mask)
            boxVal /= 2
        
        mask_fg = (bbox_full >= boxVal).type(torch.uint8)
        prediction_fg = torch.mul(prediction_full, mask_fg)
        #pred_fg_ratio = prediction_fg.sum() / (prediction_full.sum() + 1e-1)
        unionsection_full += prediction_full.sum() - prediction_fg.sum()

        #loss_sum = loss_sum + (1 - (iou_full / (weight_full + 1e-1) * pred_fg_ratio))
        loss_sum += 1 - intersection_full / (unionsection_full + 1e-1)
        #loss_sum = loss_sum + (1 - (iou_full / (weight_full + 1e-1) + pred_fg_ratio) / 2)
    return loss_sum / predictions.shape[0]


def Label_loss(predictions, labels, flow_mul):
    assert predictions.shape[0] == labels.shape[0]

    loss_list = torch.zeros_like(flow_mul)
    for op_idx in range(predictions.shape[0]):
        prediction = predictions[op_idx][0]
        
        label = labels[op_idx][0]
        weight = 2 * abs(label - 0.5) * 0.5 + 0.5 
        label = 1 / (1 + torch.exp(-20 * (label - 0.5)))  #first train enhance
        #criterion = nn.L1Loss(size_average=True)
        #loss = criterion(prediction, label)
        #loss =(abs(prediction - label) * weight).sum() / (weight.sum() + 1e-5)
        loss =abs(prediction - label).sum() / (prediction.shape[0] * prediction.shape[1])
        
        loss_list[op_idx] = loss
    loss_mul = loss_list * (flow_mul * 0.5 + 0.5)#video:1 image:0.5
    return torch.sum(loss_mul) / predictions.shape[0]

def Attention_Label_loss(att_flow, labels, flow_mul):
    assert att_flow.shape[0] == labels.shape[0]

    loss_list = torch.zeros_like(flow_mul)
    for op_idx in range(att_flow.shape[0]):
        att = att_flow[op_idx][0]

        label = labels[op_idx][0]
        weight = 2 * abs(label - 0.5) * 0.5 + 0.5
        label = 1 / (1 + torch.exp(-20 * (label - 0.5)))  #first train enhance
        #criterion = nn.L1Loss(size_average=True)
        #loss = criterion(prediction, label)
        #loss =(abs(prediction - label) * weight).sum() / (weight.sum() + 1e-5)
        loss =abs(att - label).sum() / (att.shape[0] * att.shape[1])

        loss_list[op_idx] = loss
    loss_mul = loss_list * (flow_mul * 1.0 + 0.0)#video:1 image:0.0
    return torch.sum(loss_mul) / (torch.sum(flow_mul, 0) + 1e-5)

def Attention_BBox_loss(predictions, bboxs, flow_mul):
    assert predictions.shape[0] == bboxs.shape[0]

    loss_sum = 0
    for op_idx in range(predictions.shape[0]):
        if flow_mul[op_idx] <= 0:
            continue;

        bbox_full = bboxs[op_idx][0]
        prediction_full = predictions[op_idx][0]

        boxVal = 0.9
        iou_full = 0
        weight_full = 0
        intersection_full = 0
        unionsection_full = 0
        for scale in range(4):  #support max 4 objects

            mask = ((bbox_full >= boxVal) & (bbox_full < 2 * boxVal)).type(torch.uint8)
            bbox = torch.mul(bbox_full, mask)
            prediction = torch.mul(prediction_full, mask)

            if bbox.max() <= 0:
                break

            proj_x_pred, idx = torch.max(prediction, 1)
            proj_y_pred, idx = torch.max(prediction, 0)


            proj_x_bbox, idx  = torch.max(bbox, 1)
            proj_y_bbox, idx  = torch.max(bbox, 0)

            #if bbox.max() > 0:
            proj_x_bbox /= bbox.max()
            proj_y_bbox /= bbox.max()

            intersection = proj_x_pred.sum(0) * proj_y_pred.sum(0)
            unionsection = proj_x_bbox.sum(0) * proj_y_bbox.sum(0) #equal to mask.sum()
            iou_full += intersection / (unionsection + 1e-1)
            intersection_full += intersection
            boxVal /= 2

        mask_fg = (bbox_full >= boxVal).type(torch.uint8)
        mask_bg = (bbox_full < boxVal).type(torch.uint8)
        prediction_fg = torch.mul(prediction_full, mask_fg)
        prediction_bg = torch.mul(prediction_full, mask_bg)
        #pred_fg_ratio = prediction_fg.sum() / (prediction_full.sum() + 1e-1)
        #unionsection_full += prediction_full.sum() - prediction_fg.sum()
        proj_x_pred_fg, idx = torch.max(prediction_fg, 1)
        proj_y_pred_fg, idx = torch.max(prediction_fg, 0)
        proj_x_pred_bg, idx = torch.max(prediction_bg, 1)
        proj_y_pred_bg, idx = torch.max(prediction_bg, 0)

        iou_bg = intersection_full / (intersection_full + proj_x_pred_bg.sum(0) * proj_y_pred_bg.sum(0) + 1e-1)
        iou_all = iou_full / (scale + 1e-2) * iou_bg

        #loss_sum = loss_sum + (1 - (iou_full / (weight_full + 1e-1) *  pred_fg_ratio) )
        #loss_sum += 1 - intersection_full / (unionsection_full + 1e-1)
        loss_sum += 1 - iou_all
        #loss_sum = loss_sum + (1 - (iou_full / (weight_full + 1e-1) +  pred_fg_ratio) / 2) 
    return loss_sum / (torch.sum(flow_mul, 0) + 1e-5)


def Attention_loss(att_flow_lst, labels, flow_mul):
    #att_flow_lst = [att_flow_enc_conv1, att_flow_enc_layer1, att_flow_enc_layer2, att_flow_enc_layer3, att_flow_enc_layer4, att_flow_dec]
    #att_flow_weight = [0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    att_flow_weight = [0.0, 0.0, 0.0, 0.0, 0.0, 0.6]
    att_sum = 0
    for idx in range(len(att_flow_lst)):
        if att_flow_weight[idx] <= 0:
            continue
        att = Attention_Label_loss(att_flow_lst[idx], labels, flow_mul)
        #att = Attention_BBox_loss(att_flow_lst[idx], labels, flow_mul)
        att_sum += att * att_flow_weight[idx]
    return att_sum

def Label_Check_loss(predictions, pred_motions, flow_mul):
    assert predictions.shape[0] == pred_motions.shape[0]

    loss_list = torch.zeros_like(flow_mul)
    for op_idx in range(predictions.shape[0]):
        if flow_mul[op_idx] == 0:
            continue
        prediction = predictions[op_idx][0]
        pred_motion = pred_motions[op_idx][0]

        loss = torch.sum(torch.abs(prediction - pred_motion)) /  (prediction.shape[0] * prediction.shape[1])
        loss_list[op_idx] = loss
    return torch.sum(loss_list) /  (torch.sum(flow_mul, 0) + 1e-5)

def SaliencyPair_loss(mask, mask_fg, mask_bg, image, flow):
    dirs = torch.tensor([[2, 0], [0, 2]])
    loss_sum = 0
    prediction = mask_fg
    bbox = mask

    h = mask.shape[0]
    w = mask.shape[1]
    for ii, dir in enumerate(dirs):
        h2 = dir[0]
        w2 = dir[1]

        pred_dst = prediction[h2 : h,      w2 : w]
        pred_src = prediction[ 0 : h - h2,  0 : w - w2]
        edge = 1 + 2 * pred_dst * pred_src - pred_dst - pred_src  #dst * src + (1 - dst) * (1 - src)

        image_dst = image[: , h2 : h,      w2 : w]
        image_src = image[: ,  0 : h - h2,  0 : w - w2]
        flow_dst = flow[: , h2 : h,      w2 : w]
        flow_src = flow[: ,  0 : h - h2,  0 : w - w2]
        #simi3 = torch.exp(- torch.abs(image_dst - image_src) / 0.3 * 0.125  - torch.abs(flow_dst - flow_src) / 0.3 * 1.0)
        simi3 = torch.exp(- torch.abs(flow_dst - flow_src) / 0.3 * 1.0)
        simi = torch.mean(simi3, 0)

        bbox_src = bbox[ 0 : h - h2,  0 : w - w2]

        #only sel positive sample
        #sel = torch.gt(simi, 0.2).char()
        #bbox_src = torch.mul(bbox_src, sel)

        edge = (edge + 0.5e-5) / (1 + 1e-5)

        #corr = simi * (1 - edge)
        #corr = torch.mul(corr, bbox_src)
        #loss = torch.sum(corr) / (torch.sum(bbox_src) + 1e-5)

        bce = - (simi * torch.log(edge) + (1 - simi) * torch.log(1 - edge))
        bce = torch.mul(bce, bbox_src)
        loss = torch.sum(bce) / (torch.sum(bbox_src) + 1e-5)

        #criterion = nn.CrossEntropyLoss()
        #loss = criterion(simi, edge)

        loss_sum = loss_sum + loss
    return loss_sum

def SaliencyBnd_loss(mask, prediction, image_full, flow_full):
    h = prediction.shape[0]
    w = prediction.shape[1]
    mh = max(torch.max(torch.sum(mask, 0)) // 16, 2)
    mw = max(torch.max(torch.sum(mask, 1)) // 16, 2)

    crops = torch.tensor([[0, h, mw, w + mw], [2 * mh, h + 2 * mh, mw, w + mw], [mh, h + mh, 0, w], [mh, h + mh, 2 * mw, w + 2 * mw]])

    mask_fg = torch.mul(mask, (prediction >= 0.5).type(torch.uint8))
    mask_bg = torch.mul(mask, (prediction < 0.5).type(torch.uint8))

    image_sal_sel = torch.ones((h, w)).cuda() * 255
    flow_sal_sel = torch.ones((h, w)).cuda() * 255
    for ii, crop in enumerate(crops):
        mask_ext = torch.zeros((h + 2 * mh, w + 2 * mw)).cuda()
        mask_ext[mh : h + mh,      mw : w + mw] = mask
        mask_crop = mask_ext[crop[0] : crop[1],      crop[2] : crop[3]]
        if torch.sum(mask) != torch.sum(mask_crop):
            continue
        mask_bnd = torch.mul(mask_crop, 1 - mask)
        image_crop = torch.mul(image_full, mask_bnd)
        flow_crop = torch.mul(flow_full, mask_bnd)
        image_crop_aver = torch.sum(torch.sum(image_crop, -1), -1) / torch.sum(mask_bnd)
        flow_crop_aver = torch.sum(torch.sum(flow_crop, -1), -1) / torch.sum(mask_bnd)

        image_sal = torch.abs(image_full - image_crop_aver.unsqueeze(-1).unsqueeze(-1))
        image_sal = torch.mul(image_sal, mask)
        image_sal_sel = torch.min(image_sal_sel, torch.sum(image_sal, 0))
        #image_sal_sel = image_sal_sel + torch.sum(image_sal, 0)

        flow_sal = torch.abs(flow_full - flow_crop_aver.unsqueeze(-1).unsqueeze(-1))
        flow_sal = torch.mul(flow_sal, mask)
        flow_sal_sel = torch.min(flow_sal_sel, torch.sum(flow_sal, 0))
        #flow_sal_sel = flow_sal_sel + torch.sum(flow_sal, 0)

    image_sal_fg = torch.mul(image_sal_sel, mask_fg)
    image_sal_bg = torch.mul(image_sal_sel, mask_bg)
    image_sal = torch.sum(image_sal_fg) / (torch.sum(mask_fg) + 1e-5) - torch.sum(image_sal_bg) / (torch.sum(mask_bg) + 1e-5)
    image_sal /= max(torch.max(image_sal_fg), torch.max(image_sal_bg), 1e-5) #normalize to 0:1

    flow_sal_fg = torch.mul(flow_sal_sel, mask_fg)
    flow_sal_bg = torch.mul(flow_sal_sel, mask_bg)
    flow_sal = torch.sum(flow_sal_fg) / (torch.sum(mask_fg) + 1e-5) - torch.sum(flow_sal_bg) / (torch.sum(mask_bg) + 1e-5)
    flow_sal /= max(torch.max(flow_sal_fg), torch.max(flow_sal_bg), 1e-5) #normalize to 0:1

    image_sal_sel = image_sal_sel / max(torch.max(image_sal_sel), 1e-5) #normalize to 0:1
    image_sal_loss = torch.sum(torch.abs(prediction - image_sal_sel)) / torch.sum(mask)
    flow_sal_sel = flow_sal_sel / max(torch.max(flow_sal_sel), 1e-5) #normalize to 0:1
    flow_sal_loss = torch.sum(torch.abs(prediction - flow_sal_sel)) / torch.sum(mask)

    pair_loss = SaliencyPair_loss(mask, mask_fg, mask_bg, image_full, flow_full)

    return image_sal, flow_sal, image_sal_loss, flow_sal_loss, pair_loss


def Saliency_loss(images, flows, predictions, bboxs):
    assert predictions.shape[0] == bboxs.shape[0]

    loss_sum = 0
    weight_sum = 0
    for op_idx in range(predictions.shape[0]):
        image_full = images[op_idx]
        flow_full = flows[op_idx]
        bbox_full = bboxs[op_idx][0]
        prediction_full = predictions[op_idx][0]

        boxVal = 0.9
        for scale in range(4):  #support max 4 objects

            mask = ((bbox_full >= boxVal) & (bbox_full < 2 * boxVal)).type(torch.uint8)
            bbox = torch.mul(bbox_full, mask)
            prediction = torch.mul(prediction_full, mask)

            if bbox.max() <= 0:
                break

            """
            image_sal, flow_sal, image_sal_loss, flow_sal_loss, pair_loss = SaliencyBnd_loss(mask, prediction, image_full, flow_full)
            #loss_sum += (1 - (image_sal * 0.0 + flow_sal * 1.0)) * torch.sum(mask)
            #loss_sum += image_sal_loss * torch.sum(mask)
            #loss_sum += flow_sal_loss * torch.sum(mask)
            loss_sum += pair_loss * torch.sum(mask)
            weight_sum += torch.sum(mask)
            """
            boxVal /= 2

        mask_bg = (bbox_full < boxVal).type(torch.uint8)
        prediction_bg = torch.mul(prediction_full, mask_bg)
        loss_sum += torch.sum(prediction_bg)
        weight_sum += torch.sum(mask_bg)

    return loss_sum / weight_sum



def Binary_loss(predictions, labels):
    assert predictions.shape[0] == labels.shape[0]

    loss_sum = 0
    for op_idx in range(predictions.shape[0]):
        prediction = predictions[op_idx][0]

        mask0 = torch.zeros_like(prediction)
        mask1 = torch.ones_like(prediction)
        diff = (prediction - mask0) * (mask1 - prediction)
        loss =diff.sum() / (prediction.shape[0] * prediction.shape[1])

        loss_sum = loss_sum + loss
    return loss_sum / predictions.shape[0]



def Pair_loss(images, flows, flow_mul, predictions, bboxs):
    assert predictions.shape[0] == bboxs.shape[0]
    assert flows.shape[0] == bboxs.shape[0]
    assert images.shape[0] == bboxs.shape[0]
    assert predictions.shape[0] == bboxs.shape[0]

    loss_list = torch.zeros_like(flow_mul)
    dirs = torch.tensor([[2, 0], [0, 2]])
    for op_idx in range(predictions.shape[0]):
        loss_sum = 0
        image = images[op_idx]
        flow = flows[op_idx]
        prediction = predictions[op_idx][0]
        bbox = bboxs[op_idx][0] > 0

        h = prediction.shape[0]
        w = prediction.shape[1]
        for ii, dir in enumerate(dirs):
            h2 = dir[0]
            w2 = dir[1]

            pred_dst = prediction[h2 : h,      w2 : w]
            pred_src = prediction[ 0 : h - h2,  0 : w - w2]
            edge = 1 + 2 * pred_dst * pred_src - pred_dst - pred_src  #dst * src + (1 - dst) * (1 - src)

            image_dst = image[: , h2 : h,      w2 : w]
            image_src = image[: ,  0 : h - h2,  0 : w - w2]
            flow_dst = flow[: , h2 : h,      w2 : w]
            flow_src = flow[: ,  0 : h - h2,  0 : w - w2]
            #simi3 = torch.exp(- torch.abs(image_dst - image_src) / 0.3 * 0.125  - torch.abs(flow_dst - flow_src) / 0.3 * 1.0)
            simi3 = torch.exp(- torch.abs(flow_dst - flow_src) / 0.3 * 1.0)
            simi = torch.mean(simi3, 0)

            bbox_src = bbox[ 0 : h - h2,  0 : w - w2]

            #only sel positive sample
            #sel = torch.gt(simi, 0.2).char()
            #bbox_src = torch.mul(bbox_src, sel)
            
            edge = (edge + 0.5e-5) / (1 + 1e-5)

            #corr = simi * (1 - edge)
            #corr = torch.mul(corr, bbox_src)
            #loss = torch.sum(corr) / (torch.sum(bbox_src) + 1e-5)
     
            bce = - (simi * torch.log(edge) + (1 - simi) * torch.log(1 - edge))
            bce = torch.mul(bce, bbox_src)
            loss = torch.sum(bce) / (torch.sum(bbox_src) + 1e-5)
           
            #criterion = nn.CrossEntropyLoss()
            #loss = criterion(simi, edge)

            loss_sum = loss_sum + loss
        loss_list[op_idx][0] = loss_sum / dirs.shape[0]
    loss_mul = loss_list * flow_mul
    return torch.sum(loss_mul) / (torch.sum(flow_mul) + 1e-5)
    #return torch.sum(loss_list) / loss_list.shape[0]


def Feat_loss(images, flows, flow_mul, predictions, bboxs, feats):
    assert predictions.shape[0] == bboxs.shape[0]
    assert flows.shape[0] == bboxs.shape[0]
    assert images.shape[0] == bboxs.shape[0]
    assert predictions.shape[0] == bboxs.shape[0]
    assert(predictions.shape[2] == feats.shape[2] * 4 and predictions.shape[3] == feats.shape[3] * 4)
    #feats_zoom = F.interpolate(feats, size=(predictions.shape[2], predictions.shape[3]), mode='nearest')
    #assert(predictions.shape[2] == feats_zoom.shape[2] and predictions.shape[3] == feats_zoom.shape[3])

    loss_list = torch.zeros_like(flow_mul)
    dirs = torch.tensor([[predictions.shape[3] // 20, 0], [0, predictions.shape[2] // 20]])
    for op_idx in range(predictions.shape[0]):
        loss_sum = 0
        prediction = predictions[op_idx][0]
        feat = feats[op_idx]
        #feat = torch.unsqueeze(feat, 0)
        #feat = F.interpolate(feat, size=(predictions.shape[2], predictions.shape[3]), mode='nearest')
        #assert(predictions.shape[2] == feat.shape[2] and predictions.shape[3] == feat.shape[3])
        #feat=torch.squeeze(feat, 0)
        prediction = torch.unsqueeze(prediction, 0)
        prediction = torch.unsqueeze(prediction, 0)
        prediction = F.interpolate(prediction, size=(feat.shape[1], feat.shape[2]), mode='nearest')
        assert(prediction.shape[2] == feat.shape[1] and prediction.shape[3] == feat.shape[2])
        prediction=torch.squeeze(prediction, 0)
        prediction=torch.squeeze(prediction, 0)

        #feat= torch.range(0,11)
        #feat=feat.reshape(3,2,2)
        #feat=feat.reshape(3, -1, 1)
        #feat =feat.squeeze(-1)

        area = feat.shape[1] * feat.shape[2]
        index = torch.LongTensor(random.sample(range(area), area // 4)).cuda()

        feat=feat.reshape(304, -1, 1)
        feat =feat.squeeze(-1)
        feat = torch.index_select(feat, 1, index)
        feat_mat_mut=feat.t().mm(feat)
        feat_norm1 = torch.norm(feat, dim=0).unsqueeze(0)
        feat_mat_self=feat_norm1.t().mm(feat_norm1)
        simi = (feat_mat_mut + 0.5e-5) / (feat_mat_self + 0.5e-5) / 2 + 0.5
        #simi = (cos + 1) / 2

        pred=prediction.reshape(-1, 1)
        pred = torch.index_select(pred, 0, index)
        preI = 1 - pred
        pred_mat_mut=pred.mm(pred.t())      
        preI_mat_mut=preI.mm(preI.t())
        edge = (pred_mat_mut + preI_mat_mut + 0.5e-5) / (1 + 1e-5)  #dst * src + (1 - dst) * (1 - src)
        bce = - (simi * torch.log(edge) + (1 - simi) * torch.log(1 - edge))
        #bce = torch.abs(simi - pred_mat_mut - preI_mat_mut) 
        loss_list[op_idx][0] = torch.mean(bce)
        """
        h = prediction.shape[0]
        w = prediction.shape[1]
        for ii, dir in enumerate(dirs):
            h2 = dir[0]
            w2 = dir[1]

            pred_dst = prediction[h2 : h,      w2 : w]
            pred_src = prediction[ 0 : h - h2,  0 : w - w2]
            edge = 1 + 2 * pred_dst * pred_src - pred_dst - pred_src  #dst * src + (1 - dst) * (1 - src)

            feat_dst = feat[: , h2 : h,      w2 : w]
            feat_src = feat[: ,  0 : h - h2,  0 : w - w2]

            simi = torch.sum(feat_src*feat_dst,0)/(torch.norm(feat_src, dim=0)*torch.norm(feat_dst, dim=0)+1e-5)  #xTy/(|x| * |y|)

            edge = (edge + 0.5e-5) / (1 + 1e-5)

            bce = - (simi * torch.log(edge) + (1 - simi) * torch.log(1 - edge))
            loss = torch.mean(bce)

            loss_sum = loss_sum + loss
        loss_list[op_idx][0] = loss_sum / dirs.shape[0]
        """
    loss_mul = loss_list * flow_mul
    return torch.sum(loss_mul) / (torch.sum(flow_mul) + 1e-5)
    #return torch.sum(loss_list) / loss_list.shape[0]



def lr_poly(base_lr, iter_, max_iter=100, power=0.9):
	return base_lr * ((1 - float(iter_) / max_iter) ** power)

def get_mae(preds, labels):
	assert(preds.numel() == labels.numel())
	if preds.dim() == labels.dim():
		mae = torch.mean(torch.abs(preds - labels))
		return mae.item()
	else:
		mae = torch.mean(torch.abs(preds.squeeze() - labels.squeeze()))
		return mae.item()

def get_prec_recall(preds, labels):
	assert(preds.numel() == labels.numel())
	preds_ = preds.cpu().data.numpy()
	labels_ = labels.cpu().data.numpy()
	preds_ = preds_.squeeze(1)
	labels_ = labels_.squeeze(1)
	assert(len(preds_.shape) == len(labels_.shape)) 
	assert(len(preds_.shape) == 3)
	prec_list = []
	recall_list = []

	assert(preds_.min() >= 0 and preds_.max() <= 1)
	assert(labels_.min() >= 0 and labels_.max() <= 1)
	for i in range(preds_.shape[0]):
		pred_, label_ = preds_[i], labels_[i]
		thres_ = pred_.sum() * 2.0 / pred_.size

		binari_ = np.zeros(shape=pred_.shape, dtype=np.uint8)
		binari_[np.where(pred_ >= thres_)] = 1

		label_ = label_.astype(np.uint8)
		matched_ = np.multiply(binari_, label_)

		TP = matched_.sum()
		TP_FP = binari_.sum()
		TP_FN = label_.sum()
		prec = (TP + 1e-6) / (TP_FP + 1e-6)
		recall = (TP + 1e-6) / (TP_FN + 1e-6)
		prec_list.append(prec) 
		recall_list.append(recall)
	return prec_list, recall_list

def decode_segmap(label_mask):
	r = label_mask.copy()
	g = label_mask.copy()
	b = label_mask.copy()
	rgb = np.zeros((label_mask.shape[0], label_mask.shape[1], 3))
	rgb[:, :, 0] = r * 255.0
	rgb[:, :, 1] = g * 255.0
	rgb[:, :, 2] = b * 255.0
	rgb = np.rint(rgb).astype(np.uint8)
	return rgb

def decode_seg_map_sequence(label_masks):
	assert(label_masks.ndim == 3 or label_masks.ndim == 4)
	if label_masks.ndim == 4:
		label_masks = label_masks.squeeze(1)
	assert(label_masks.ndim == 3)
	rgb_masks = []
	for i in range(label_masks.shape[0]):
		rgb_mask = decode_segmap(label_masks[i])
		rgb_masks.append(rgb_mask)
	rgb_masks = torch.from_numpy(np.array(rgb_masks).transpose([0, 3, 1, 2]))
	return rgb_masks
