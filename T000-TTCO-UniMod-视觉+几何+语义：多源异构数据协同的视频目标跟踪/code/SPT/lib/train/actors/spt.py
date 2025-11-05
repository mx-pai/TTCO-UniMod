from . import BaseActor
from lib.utils.misc import NestedTensor
from lib.utils.box_ops import box_cxcywh_to_xyxy, box_xywh_to_xyxy
import torch
import torch.nn.functional as F
from lib.utils.merge import merge_template_search


class SPTActor(BaseActor):
    """ Actor for training the multi-modal SPT"""
    def __init__(self, net, objective, loss_weight, settings):
        super().__init__(net, objective)
        self.loss_weight = loss_weight
        self.settings = settings
        self.bs = self.settings.batchsize  # batch size
        self.lang_loss_weight = getattr(settings, 'lang_loss_weight', 0.0)

    def __call__(self, data):
        """
        args:
            data - The input data, should contain the fields 'template', 'search', 'gt_bbox'.
            template_images: (N_t, batch, 3, H, W)
            search_images: (N_s, batch, 3, H, W)
        returns:
            loss    - the training loss
            status  -  dict containing detailed losses
        """
        # forward pass
        out_dict = self.forward_pass(data, run_box_head=True, run_cls_head=False)

        # prepare gt bboxes
        gt_bboxes = data['search_anno']

        if isinstance(out_dict, list):
            gt_seq = [gt_bboxes[i] for i in range(gt_bboxes.shape[0])]
            loss, status = self.compute_sequence_losses(out_dict, gt_seq)
            lang_out_list = out_dict
        else:
            loss, status = self.compute_losses(out_dict, gt_bboxes[0])
            lang_out_list = [out_dict]

        if self.lang_loss_weight > 0:
            loss, status = self._apply_language_loss(lang_out_list, data, loss, status)

        return loss, status

    def forward_pass(self, data, run_box_head, run_cls_head, text_data=None):
        if text_data is None:
            text_ids = data['nl_token_ids'].permute(1, 0)
            text_masks = data['nl_token_masks'].permute(1, 0)
            text_data = NestedTensor(text_ids, text_masks)
        text_dict = self.net(text_data=text_data, mode="language_backbone")

        # encode templates once
        template_color_feats = []
        template_depth_feats = []
        for i in range(self.settings.num_template):
            template_img = data['template_images'][i].view(-1, *data['template_images'].shape[2:])
            template_att = data['template_att'][i].view(-1, *data['template_att'].shape[2:])
            template_color = self.net(img=NestedTensor(template_img[:, :3], template_att), mode='backbone_color')
            template_depth = self.net(img=NestedTensor(template_img[:, 3:], template_att), mode='backbone_depth')
            template_color_feats.append(template_color)
            template_depth_feats.append(template_depth)

        num_search = data['search_images'].shape[0] if data['search_images'].dim() >= 5 else 1
        outputs = []

        for idx in range(num_search):
            search_img = data['search_images'][idx].view(-1, *data['search_images'].shape[2:])
            search_att = data['search_att'][idx].view(-1, *data['search_att'].shape[2:])
            search_color = self.net(img=NestedTensor(search_img[:, :3], search_att), mode='backbone_color')
            search_depth = self.net(img=NestedTensor(search_img[:, 3:], search_att), mode='backbone_depth')

            color_feat_dict_list = template_color_feats + [search_color]
            depth_feat_dict_list = template_depth_feats + [search_depth]
            visiontext_feat_dict_list = [text_dict] + template_color_feats + [search_color]

            seq_dict_color = merge_template_search(color_feat_dict_list)
            seq_dict_depth = merge_template_search(depth_feat_dict_list)
            seq_dict_vl = merge_template_search(visiontext_feat_dict_list)

            out_dict, _, _ = self.net(seq_dict_c=seq_dict_color, seq_dict_d=seq_dict_depth, seq_dict_vl=seq_dict_vl,
                                      mode="transformer", run_box_head=run_box_head, run_cls_head=run_cls_head)
            outputs.append(out_dict)

        return outputs[0] if len(outputs) == 1 else outputs

    def compute_losses(self, pred_dict, gt_bbox, return_status=True):
        # Get boxes
        pred_boxes = pred_dict['pred_boxes']
        if torch.isnan(pred_boxes).any():
            raise ValueError("Network outputs is NAN! Stop Training")
        num_queries = pred_boxes.size(1)
        pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)  # (B,N,4) --> (BN,4) (x1,y1,x2,y2)
        gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0, max=1.0)  # (B,4) --> (B,1,4) --> (B,N,4)
        # compute giou and iou
        try:
            giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        except:
            giou_loss, iou = torch.tensor(0.0).cuda(), torch.tensor(0.0).cuda()
        # compute l1 loss
        l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)  # (BN,4) (BN,4)
        # weighted sum
        loss = self.loss_weight['giou'] * giou_loss + self.loss_weight['l1'] * l1_loss
        if return_status:
            # status for log
            mean_iou = iou.detach().mean()
            status = {"Loss/total": loss.item(),
                      "Loss/giou": giou_loss.item(),
                      "Loss/l1": l1_loss.item(),
                      "IoU": mean_iou.item()}
            return loss, status
        else:
            return loss

    def compute_sequence_losses(self, pred_dicts, gt_bboxes_seq):
        if len(pred_dicts) == 0:
            raise ValueError("Empty prediction list for sequence loss computation.")

        device = pred_dicts[0]['pred_boxes'].device
        total_giou = torch.zeros(1, device=device)
        total_l1 = torch.zeros(1, device=device)
        total_iou = torch.zeros(1, device=device)
        count = 0

        for pred_dict, gt_bbox in zip(pred_dicts, gt_bboxes_seq):
            pred_boxes = pred_dict['pred_boxes']
            if torch.isnan(pred_boxes).any():
                raise ValueError("Network outputs is NAN! Stop Training")
            num_queries = pred_boxes.size(1)
            pred_boxes_vec = box_cxcywh_to_xyxy(pred_boxes).view(-1, 4)
            gt_boxes_vec = box_xywh_to_xyxy(gt_bbox)[:, None, :].repeat((1, num_queries, 1)).view(-1, 4).clamp(min=0.0, max=1.0)

            try:
                giou_loss, iou = self.objective['giou'](pred_boxes_vec, gt_boxes_vec)
            except:
                giou_loss, iou = torch.tensor(0.0, device=device), torch.tensor(0.0, device=device)
            l1_loss = self.objective['l1'](pred_boxes_vec, gt_boxes_vec)

            total_giou = total_giou + giou_loss
            total_l1 = total_l1 + l1_loss
            total_iou = total_iou + iou.detach().mean()
            count += 1

        avg_giou = total_giou / count
        avg_l1 = total_l1 / count
        avg_iou = total_iou / count
        loss = self.loss_weight['giou'] * avg_giou + self.loss_weight['l1'] * avg_l1

        status = {"Loss/total": loss.item(),
                  "Loss/giou": avg_giou.item(),
                  "Loss/l1": avg_l1.item(),
                  "IoU": avg_iou.item(),
                  "SeqLen": float(count)}
        return loss, status

    def _apply_language_loss(self, out_dict_list, data, loss, status):
        lang_scores_pos = []
        for out in out_dict_list:
            if 'lang_score' in out:
                lang_scores_pos.append(out['lang_score'].view(out['lang_score'].size(0), -1))

        if not lang_scores_pos:
            status["Loss/lang"] = 0.0
            return loss, status

        lang_score_pos = torch.stack(lang_scores_pos, dim=0).mean(dim=0).view(-1, 1)
        batch_size = lang_score_pos.size(0)
        status["LangScore/pos"] = lang_score_pos.mean().item()

        if batch_size > 1:
            perm = torch.arange(batch_size, device=lang_score_pos.device)
            perm = torch.roll(perm, shifts=1)
            neg_ids = data['nl_token_ids'][perm]
            neg_masks = data['nl_token_masks'][perm]
            text_ids_neg = neg_ids.permute(1, 0)
            text_masks_neg = neg_masks.permute(1, 0)
            text_data_neg = NestedTensor(text_ids_neg, text_masks_neg)
            out_dict_neg = self.forward_pass(data, run_box_head=False, run_cls_head=False, text_data=text_data_neg)
            if isinstance(out_dict_neg, list):
                lang_scores_neg_list = [od['lang_score'].view(od['lang_score'].size(0), -1)
                                        for od in out_dict_neg if 'lang_score' in od]
                if lang_scores_neg_list:
                    lang_score_neg = torch.stack(lang_scores_neg_list, dim=0).mean(dim=0).view(-1, 1)
                else:
                    lang_score_neg = torch.zeros_like(lang_score_pos)
            else:
                if 'lang_score' in out_dict_neg:
                    lang_score_neg = out_dict_neg['lang_score'].view(-1, 1)
                else:
                    lang_score_neg = torch.zeros_like(lang_score_pos)

            status["LangScore/neg"] = lang_score_neg.mean().item()

            lang_scores = torch.cat([lang_score_pos, lang_score_neg], dim=0)
            lang_labels = torch.cat([torch.ones_like(lang_score_pos),
                                     torch.zeros_like(lang_score_neg)], dim=0)
            lang_loss = F.binary_cross_entropy(lang_scores, lang_labels)
            loss = loss + self.lang_loss_weight * lang_loss
            status["Loss/lang"] = lang_loss.item()
        else:
            status["Loss/lang"] = 0.0

        return loss, status
