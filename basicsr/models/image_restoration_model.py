import importlib
import torch
from torch.nn import MSELoss
from collections import OrderedDict
from copy import deepcopy
from os import path as osp
from tqdm import tqdm
from hrseg.hrseg_model import create_hrnet
# from basicsr.models.archs.DRSformer_arch import getsegmap
from basicsr.models.archs import define_network
from basicsr.models.base_model import BaseModel
from basicsr.utils import get_root_logger, imwrite, tensor2img
from basicsr.models.losses.CR_loss import Vgg19,ContrastLoss
from basicsr.criteria.lpips.lpips import LPIPS1 as LPIPS
loss_module = importlib.import_module('basicsr.models.losses')
metric_module = importlib.import_module('basicsr.metrics')
# from basicsr.models.losses.SKF_loss import CharbonnierLoss,  hist_loss
import os
import random
import numpy as np
import cv2
import random
from PIL import Image
import torch.nn.functional as F
from functools import partial
import matplotlib.pyplot as plt

class Mixing_Augment:
    def __init__(self, mixup_beta, use_identity, device):
        self.dist = torch.distributions.beta.Beta(torch.tensor([mixup_beta]), torch.tensor([mixup_beta]))
        self.device = device

        self.use_identity = use_identity

        self.augments = [self.mixup]

    def mixup(self, target, input_):
        lam = self.dist.rsample((1,1)).item()
    
        r_index = torch.randperm(target.size(0)).to(self.device)
    
        target = lam * target + (1-lam) * target[r_index, :]
        input_ = lam * input_ + (1-lam) * input_[r_index, :]
    
        return target, input_

    def __call__(self, target, input_):
        if self.use_identity:
            augment = random.randint(0, len(self.augments))
            if augment < len(self.augments):
                target, input_ = self.augments[augment](target, input_)
        else:
            augment = random.randint(0, len(self.augments)-1)
            target, input_ = self.augments[augment](target, input_)
        return target, input_
def cross_entropy_loss_RCF(prediction, labelf, beta):
    label = labelf.long()
    mask = labelf.clone()
    num_positive = torch.sum(label==1).float()
    num_negative = torch.sum(label==0).float()

    mask[label == 1] = 1.0 * num_negative / (num_positive + num_negative)
    mask[label == 0] = beta * num_positive / (num_positive + num_negative)
    mask[label == 2] = 0
    cost = F.binary_cross_entropy(prediction, labelf, weight=mask, reduction='mean')
    return cost
def flip(x, dim):
	indices = [slice(None)] * x.dim()
	indices[dim] = torch.arange(x.size(dim) - 1, -1, -1, dtype=torch.long, device=x.device)
	return x[tuple(indices)]
def get_sketch(img):   
    imgs = torch.unbind(img, dim=0)
    sketch_list = []
    for i ,img in enumerate(imgs):
        img = img.permute(1,2,0)
        to_im = img.cpu().numpy()
        to_im_gray = cv2.cvtColor(to_im, cv2.COLOR_BGR2GRAY)
        sketch = cv2.GaussianBlur(to_im_gray, (3, 3), 0)
 
        v = np.median(sketch)
        sigma = 0.33
        lower = int(max(0, (1.0 - sigma) * v))
        upper = int(min(255, (1.0 + sigma) * v))
        sketch = sketch.astype(np.uint8)
        sketch = cv2.Canny(sketch, lower, upper)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        sketch = cv2.dilate(sketch, kernel)

        sketch = np.expand_dims(sketch, axis=-1)
        sketch = np.concatenate([sketch, sketch, sketch], axis=-1)
        assert len(np.unique(sketch)) == 2 or len(np.unique(sketch)) == 1
        to_im = to_im[:, :, [2,1,0]]
        to_im = to_im.astype(np.uint8)
        to_im = Image.fromarray(to_im)
        to_im = np.array(to_im)
        to_im=(to_im+1)*0.5
        height = to_im.shape[0]
        width = to_im.shape[1]
        sketch[sketch == 255] = 1
        sketch = cv2.resize(sketch, (width, height))
        sketch = torch.from_numpy(sketch).permute(2, 0, 1)
        sketch = sketch[0:1, :, :]
        sketch = sketch.long()
        sketch_list.append(sketch)
    sketch = torch.stack(sketch_list).float()
    return sketch
class ImageCleanModel(BaseModel):
    """Base Deblur model for single image deblur."""

    def __init__(self, opt):
        super(ImageCleanModel, self).__init__(opt)

        # define network

        self.mixing_flag = self.opt['train']['mixing_augs'].get('mixup', False)
        if self.mixing_flag:
            mixup_beta       = self.opt['train']['mixing_augs'].get('mixup_beta', 1.2)
            use_identity     = self.opt['train']['mixing_augs'].get('use_identity', False)
            self.mixing_augmentation = Mixing_Augment(mixup_beta, use_identity, self.device)

        self.net_g = define_network(deepcopy(opt['network_g']))
        self.net_g = self.model_to_device(self.net_g)
        # self.seg_model = create_hrnet().cuda()
        self.print_network(self.net_g)

        # load pretrained models
        load_path = self.opt['path'].get('pretrain_network_g', None)
        if load_path is not None:
            self.load_network(self.net_g, load_path,
                              self.opt['path'].get('strict_load_g', True), param_key=self.opt['path'].get('param_key', 'params'))

        if self.is_train:
            self.init_training_settings()

    def init_training_settings(self):
        self.net_g.train()
        train_opt = self.opt['train']

        self.ema_decay = train_opt.get('ema_decay', 0)
        if self.ema_decay > 0:
            logger = get_root_logger()
            logger.info(
                f'Use Exponential Moving Average with decay: {self.ema_decay}')
            # define network net_g with Exponential Moving Average (EMA)
            # net_g_ema is used only for testing on one GPU and saving
            # There is no need to wrap with DistributedDataParallel
            self.net_g_ema = define_network(self.opt['network_g']).to(
                self.device)
            # load pretrained model
            load_path = self.opt['path'].get('pretrain_network_g', None)
            if load_path is not None:
                self.load_network(self.net_g_ema, load_path,
                                  self.opt['path'].get('strict_load_g',
                                                       True), 'params_ema')
            else:
                self.model_ema(0)  # copy net_g weight
            self.net_g_ema.eval()

        # define losses
        if train_opt.get('pixel_opt'):
            pixel_type = train_opt['pixel_opt'].pop('type')
            cri_pix_cls = getattr(loss_module, pixel_type)
            self.cri_pix = cri_pix_cls(**train_opt['pixel_opt']).to(
                self.device)
            self.CR_loss = ContrastLoss().to(
                self.device)
            self.lpips_loss = LPIPS().to(
                self.device)
            self.mse_loss = torch.nn.MSELoss().to(
                self.device)
            # self.edge_loss = cross_entropy_loss_RCF().to(
            #     self.device)
        else:
            raise ValueError('pixel loss are None.')

        # set up optimizers and schedulers
        self.setup_optimizers()
        self.setup_schedulers()

    def setup_optimizers(self):
        train_opt = self.opt['train']
        optim_params = []

        for k, v in self.net_g.named_parameters():
            if v.requires_grad:
                optim_params.append(v)
            else:
                logger = get_root_logger()
                logger.warning(f'Params {k} will not be optimized.')

        optim_type = train_opt['optim_g'].pop('type')
        if optim_type == 'Adam':
            self.optimizer_g = torch.optim.Adam(optim_params, **train_opt['optim_g'])
        elif optim_type == 'AdamW':
            self.optimizer_g = torch.optim.AdamW(optim_params, **train_opt['optim_g'])
        else:
            raise NotImplementedError(
                f'optimizer {optim_type} is not supperted yet.')
        self.optimizers.append(self.optimizer_g)

    def feed_train_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

        if self.mixing_flag:
            self.gt, self.lq = self.mixing_augmentation(self.gt, self.lq)

    def feed_data(self, data):
        self.lq = data['lq'].to(self.device)
        if 'gt' in data:
            self.gt = data['gt'].to(self.device)

    def optimize_parameters(self, current_iter):
        self.optimizer_g.zero_grad()
        # _ , seg_feature = self.seg_model(self.lq[:,0:3,:,:].cuda())
        preds , preds_inter , preds_sketch  ,fake_pred = self.net_g(self.lq)
        if not isinstance(preds, list):
            preds = [preds]
        sketch = get_sketch(self.gt)
        sketch = sketch.to(self.device) 
        self.output = preds[-1]
        # self.seg_map = getsegmap()
        

        loss_dict = OrderedDict()
        # pixel loss
        l_pix = 0.
        rec_loss = 0.
        cr_loss = 0.
        lpips_loss = 0.
        edge_loss = 0. 
        discri_loss=0.
        # print(len(preds))
        for pred in preds:
            rec_loss = self.mse_loss(self.gt,pred)*10 + self.mse_loss(self.gt,preds_inter)
            lpips_loss = self.lpips_loss(self.gt,pred) + self.lpips_loss(self.gt,preds_inter)
            # print('pred', pred.shape)
            # print('preds', preds.shape)
            edge_loss = cross_entropy_loss_RCF(preds_sketch,sketch,1.1)*5
            discri_loss = F.softplus(-fake_pred).mean()
            cr_loss = 0.1*self.CR_loss(pred,self.gt,self.lq)
            # l_pix = 10*self.cri_pix(pred, self.gt) + self.cri_pix(preds_inter,self.gt)

        l_pix = rec_loss + 0.8*lpips_loss + discri_loss + edge_loss + cr_loss
        loss_dict['l_pix'] = l_pix

        l_pix.backward()
        if self.opt['train']['use_grad_clip']:
            torch.nn.utils.clip_grad_norm_(self.net_g.parameters(), 0.01)
        self.optimizer_g.step()

        self.log_dict = self.reduce_loss_dict(loss_dict)

        if self.ema_decay > 0:
            self.model_ema(decay=self.ema_decay)

    def pad_test(self, window_size):        
        scale = self.opt.get('scale', 1)
        mod_pad_h, mod_pad_w = 0, 0
        _, _, h, w = self.lq.size()
        if h % window_size != 0:
            mod_pad_h = window_size - h % window_size
        if w % window_size != 0:
            mod_pad_w = window_size - w % window_size
        # print(self.lq.shape)
        img = F.pad(self.lq, (0, mod_pad_w, 0, mod_pad_h), 'reflect')
        # print(img.shape)
        # _ , seg_feature = self.seg_model(img[:,0:3,:,:].cuda())
        self.nonpad_test(img)
        _, _, h, w = self.output.size()
        self.output = self.output[:, :, 0:h - mod_pad_h * scale, 0:w - mod_pad_w * scale]
        # print(self.output.shape)

    def nonpad_test(self, img=None ):
        if img is None:
            img = self.lq      
        if hasattr(self, 'net_g_ema'):
            self.net_g_ema.eval()
            with torch.no_grad():
                preds , preds_inter , preds_sketch  ,fake_pred = self.net_g_ema(img)
            if isinstance(preds, list):
                pred = preds[-1]
            self.output = preds
        else:
            self.net_g.eval()
            with torch.no_grad():
                preds , preds_inter , preds_sketch  ,fake_pred = self.net_g(img)
            if isinstance(preds, list):
                pred = preds[-1]
            self.output = preds
            self.net_g.train()

    def dist_validation(self, dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image):
        if os.environ['LOCAL_RANK'] == '0':
            return self.nondist_validation(dataloader, current_iter, tb_logger, save_img, rgb2bgr, use_image)
        else:
            return 0.

    def nondist_validation(self, dataloader, current_iter, tb_logger,
                           save_img, rgb2bgr, use_image):
        dataset_name = dataloader.dataset.opt['name']
        with_metrics = self.opt['val'].get('metrics') is not None
        if with_metrics:
            self.metric_results = {
                metric: 0
                for metric in self.opt['val']['metrics'].keys()
            }
        # pbar = tqdm(total=len(dataloader), unit='image')

        window_size = self.opt['val'].get('window_size', 0)

        if window_size:
            test = partial(self.pad_test, window_size)
        else:
            test = self.nonpad_test

        cnt = 0

        for idx, val_data in enumerate(dataloader):
            img_name = osp.splitext(osp.basename(val_data['lq_path'][0]))[0]

            self.feed_data(val_data)
            test()

            visuals = self.get_current_visuals()
            sr_img = tensor2img([visuals['result']], rgb2bgr=rgb2bgr)
            if 'gt' in visuals:
                gt_img = tensor2img([visuals['gt']], rgb2bgr=rgb2bgr)
                del self.gt

            # tentative for out of GPU memory
            del self.lq
            del self.output
            torch.cuda.empty_cache()

            if save_img:
                
                if self.opt['is_train']:
                    
                    save_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}.png')
                    
                    save_gt_img_path = osp.join(self.opt['path']['visualization'],
                                             img_name,
                                             f'{img_name}_{current_iter}_gt.png')
                else:
                    
                    save_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}.png')
                    save_gt_img_path = osp.join(
                        self.opt['path']['visualization'], dataset_name,
                        f'{img_name}_gt.png')
                    
                imwrite(sr_img, save_img_path)
                imwrite(gt_img, save_gt_img_path)

            if with_metrics:
                # calculate metrics
                opt_metric = deepcopy(self.opt['val']['metrics'])
                if use_image:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(sr_img, gt_img, **opt_)
                else:
                    for name, opt_ in opt_metric.items():
                        metric_type = opt_.pop('type')
                        self.metric_results[name] += getattr(
                            metric_module, metric_type)(visuals['result'], visuals['gt'], **opt_)

            cnt += 1

        current_metric = 0.
        if with_metrics:
            for metric in self.metric_results.keys():
                self.metric_results[metric] /= cnt
                current_metric = self.metric_results[metric]

            self._log_validation_metric_values(current_iter, dataset_name,
                                               tb_logger)
        return current_metric


    def _log_validation_metric_values(self, current_iter, dataset_name,
                                      tb_logger):
        log_str = f'Validation {dataset_name},\t'
        for metric, value in self.metric_results.items():
            log_str += f'\t # {metric}: {value:.4f}'
        logger = get_root_logger()
        logger.info(log_str)
        if tb_logger:
            for metric, value in self.metric_results.items():
                tb_logger.add_scalar(f'metrics/{metric}', value, current_iter)

    def get_current_visuals(self):
        out_dict = OrderedDict()
        out_dict['lq'] = self.lq.detach().cpu()
        out_dict['result'] = self.output.detach().cpu()
        if hasattr(self, 'gt'):
            out_dict['gt'] = self.gt.detach().cpu()
        return out_dict

    def save(self, epoch, current_iter):
        if self.ema_decay > 0:
            self.save_network([self.net_g, self.net_g_ema],
                              'net_g',
                              current_iter,
                              param_key=['params', 'params_ema'])
        else:
            self.save_network(self.net_g, 'net_g', current_iter)
        self.save_training_state(epoch, current_iter)
