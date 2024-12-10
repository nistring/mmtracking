# Copyright (c) OpenMMLab. All rights reserved.
from abc import ABCMeta, abstractmethod
from collections import OrderedDict

import mmcv
import numpy as np
import torch
import torch.distributed as dist
from mmcv.runner import BaseModule, auto_fp16

from mmtrack.utils import get_root_logger
from typing import List, Optional, Union
import cv2
from mmcv.image import imread, imwrite


class BaseVideoDetector(BaseModule, metaclass=ABCMeta):
    """Base class for video object detector.

    Args:
        init_cfg (dict or list[dict], optional): Initialization config dict.
    """

    def __init__(self, init_cfg):
        super(BaseVideoDetector, self).__init__(init_cfg)
        self.logger = get_root_logger()
        self.fp16_enabled = False

    def freeze_module(self, module):
        """Freeze module during training."""
        if isinstance(module, str):
            modules = [module]
        else:
            if not (isinstance(module, list) or isinstance(module, tuple)):
                raise TypeError('module must be a str or a list.')
            else:
                modules = module
        for module in modules:
            m = getattr(self, module)
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    @property
    def with_detector(self):
        """bool: whether the framework has a detector"""
        return hasattr(self, 'detector') and self.detector is not None

    @property
    def with_motion(self):
        """bool: whether the framework has a motion model"""
        return hasattr(self, 'motion') and self.motion is not None

    @property
    def with_aggregator(self):
        """bool: whether the framework has a aggregator"""
        return hasattr(self, 'aggregator') and self.aggregator is not None

    @abstractmethod
    def forward_train(self,
                      imgs,
                      img_metas,
                      ref_img=None,
                      ref_img_metas=None,
                      **kwargs):
        """
        Args:
            img (Tensor): of shape (N, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.

            img_metas (list[dict]): list of image info dict where each dict
                has: 'img_shape', 'scale_factor', 'flip', and may also contain
                'filename', 'ori_shape', 'pad_shape', and 'img_norm_cfg'.
                For details on the values of these keys see
                `mmtrack/datasets/pipelines/formatting.py:VideoCollect`.

            ref_img (Tensor): of shape (N, R, C, H, W) encoding input images.
                Typically these should be mean centered and std scaled.
                R denotes there is #R reference images for each input image.

            ref_img_metas (list[list[dict]]): The first list only has one
                element. The second list contains reference image information
                dict where each dict has: 'img_shape', 'scale_factor', 'flip',
                and may also contain 'filename', 'ori_shape', 'pad_shape', and
                'img_norm_cfg'. For details on the values of these keys see
                `mmtrack/datasets/pipelines/formatting.py:VideoCollect`.
        """
        pass

    @abstractmethod
    def simple_test(self, img, img_metas, **kwargs):
        pass

    def aug_test(self, imgs, img_metas, **kwargs):
        """Test function with test time augmentation."""
        pass

    def forward_test(self,
                     imgs,
                     img_metas,
                     ref_img=None,
                     ref_img_metas=None,
                     **kwargs):
        """
        Args:
            imgs (List[Tensor]): the outer list indicates test-time
                augmentations and inner Tensor should have a shape NxCxHxW,
                which contains all images in the batch.

            img_metas (List[List[dict]]): the outer list indicates test-time
                augs (multiscale, flip, etc.) and the inner list indicates
                images in a batch.

            ref_img (list[Tensor] | None): The list only contains one Tensor
                of shape (1, N, C, H, W) encoding input reference images.
                Typically these should be mean centered and std scaled. N
                denotes the number for reference images. There may be no
                reference images in some cases.

            ref_img_metas (list[list[list[dict]]] | None): The first and
                second list only has one element. The third list contains
                image information dict where each dict has: 'img_shape',
                'scale_factor', 'flip', and may also contain 'filename',
                'ori_shape', 'pad_shape', and 'img_norm_cfg'. For details on
                the values of these keys see
                `mmtrack/datasets/pipelines/formatting.py:VideoCollect`. There
                may be no reference images in some cases.
        """
        if isinstance(imgs, torch.Tensor):
            imgs = [imgs]
        elif not isinstance(imgs, list):
            raise TypeError(
                f'imgs must be a list or tensor, but got {type(imgs)}')

        assert isinstance(img_metas, list)
        if isinstance(img_metas[0], dict):
            img_metas = [img_metas]
        elif not isinstance(img_metas[0], list):
            raise TypeError(
                'img_metas must be a List[List[dict]] or List[dict]')

        num_augs = len(imgs)
        if num_augs != len(img_metas):
            raise ValueError(f'num of augmentations ({len(imgs)}) '
                             f'!= num of image meta ({len(img_metas)})')

        if num_augs == 1:
            # proposals (List[List[Tensor]]): the outer list indicates
            # test-time augs (multiscale, flip, etc.) and the inner list
            # indicates images in a batch.
            # The Tensor should have a shape Px4, where P is the number of
            # proposals.
            if 'proposals' in kwargs:
                kwargs['proposals'] = kwargs['proposals'][0]
            return self.simple_test(
                imgs[0],
                img_metas[0],
                ref_img=ref_img,
                ref_img_metas=ref_img_metas,
                **kwargs)
        else:
            assert imgs[0].size(0) == 1, 'aug test does not support ' \
                                         'inference with batch size ' \
                                         f'{imgs[0].size(0)}'
            # TODO: support test augmentation for predefined proposals
            assert 'proposals' not in kwargs
            return self.aug_test(
                imgs,
                img_metas,
                ref_img=ref_img,
                ref_img_metas=ref_img_metas,
                **kwargs)

    @auto_fp16(apply_to=('img', 'ref_img'))
    def forward(self,
                img,
                img_metas,
                ref_img=None,
                ref_img_metas=None,
                return_loss=True,
                **kwargs):
        """Calls either :func:`forward_train` or :func:`forward_test` depending
        on whether ``return_loss`` is ``True``.

        Note this setting will change the expected inputs. When
        ``return_loss=True``, img and img_meta are single-nested (i.e. Tensor
        and List[dict]), and when ``resturn_loss=False``, img and img_meta
        should be double nested (i.e.  List[Tensor], List[List[dict]]), with
        the outer list indicating test time augmentations.
        """
        if return_loss:
            return self.forward_train(
                img,
                img_metas,
                ref_img=ref_img,
                ref_img_metas=ref_img_metas,
                **kwargs)
        else:
            return self.forward_test(
                img,
                img_metas,
                ref_img=ref_img,
                ref_img_metas=ref_img_metas,
                **kwargs)

    def _parse_losses(self, losses):
        """Parse the raw outputs (losses) of the network.

        Args:
            losses (dict): Raw output of the network, which usually contain
                losses and other necessary information.

        Returns:
            tuple[Tensor, dict]: (loss, log_vars), loss is the loss tensor \
                which may be a weighted sum of all losses, log_vars contains \
                all the variables to be sent to the logger.
        """
        log_vars = OrderedDict()
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                log_vars[loss_name] = loss_value.mean()
            elif isinstance(loss_value, list):
                log_vars[loss_name] = sum(_loss.mean() for _loss in loss_value)
            else:
                raise TypeError(
                    f'{loss_name} is not a tensor or list of tensors')

        loss = sum(_value for _key, _value in log_vars.items()
                   if 'loss' in _key)

        log_vars['loss'] = loss
        for loss_name, loss_value in log_vars.items():
            # reduce loss when distributed training
            if dist.is_available() and dist.is_initialized():
                loss_value = loss_value.data.clone()
                dist.all_reduce(loss_value.div_(dist.get_world_size()))
            log_vars[loss_name] = loss_value.item()

        return loss, log_vars

    def train_step(self, data, optimizer):
        """The iteration step during training.

        This method defines an iteration step during training, except for the
        back propagation and optimizer updating, which are done in an optimizer
        hook. Note that in some complicated cases or models, the whole process
        including back propagation and optimizer updating is also defined in
        this method, such as GAN.

        Args:
            data (dict): The output of dataloader.
            optimizer (:obj:`torch.optim.Optimizer` | dict): The optimizer of
                runner is passed to ``train_step()``. This argument is unused
                and reserved.

        Returns:
            dict: It should contain at least 3 keys: ``loss``, ``log_vars``, \
                ``num_samples``.

                - ``loss`` is a tensor for back propagation, which can be a \
                weighted sum of multiple losses.
                - ``log_vars`` contains all the variables to be sent to the
                logger.
                - ``num_samples`` indicates the batch size (when the model is \
                DDP, it means the batch size on each GPU), which is used for \
                averaging the logs.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def val_step(self, data, optimizer):
        """The iteration step during validation.

        This method shares the same signature as :func:`train_step`, but used
        during val epochs. Note that the evaluation after training epochs is
        not implemented with this method, but an evaluation hook.
        """
        losses = self(**data)
        loss, log_vars = self._parse_losses(losses)

        outputs = dict(
            loss=loss, log_vars=log_vars, num_samples=len(data['img_metas']))

        return outputs

    def show_result(self,
                    img,
                    result,
                    score_thr=0.3,
                    bbox_color='green',
                    text_color='green',
                    thickness=1,
                    font_scale=0.5,
                    win_name='',
                    show=False,
                    wait_time=0,
                    out_file=None):
        """Draw `result` over `img`.

        Args:
            img (str or Tensor): The image to be displayed.
            result (dict): The results to draw over `img` det_bboxes or
                (det_bboxes, det_masks). The value of key 'det_bboxes'
                is list with length num_classes, and each element in list
                is ndarray with shape(n, 5)
                in [tl_x, tl_y, br_x, br_y, score] format.
            score_thr (float, optional): Minimum score of bboxes to be shown.
                Default: 0.3.
            bbox_color (str or tuple or :obj:`Color`): Color of bbox lines.
            text_color (str or tuple or :obj:`Color`): Color of texts.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
                Default: 0.
            show (bool): Whether to show the image.
                Default: False.
            out_file (str or None): The filename to write the image.
                Default: None.

        Returns:
            img (Tensor): Only if not `show` or `out_file`
        """
        # TODO: make it support tracking
        img = mmcv.imread(img)
        img = img.copy()
        assert isinstance(result, dict)
        bbox_results = result.get('det_bboxes', None)
        mask_results = result.get('det_masks', None)
        if isinstance(mask_results, tuple):
            mask_results = mask_results[0]  # ms rcnn
        bboxes = np.vstack(bbox_results)
        labels = [
            np.full(bbox.shape[0], i, dtype=np.int32)
            for i, bbox in enumerate(bbox_results)
        ]
        labels = np.concatenate(labels)
        # draw segmentation masks
        if mask_results is not None and len(labels) > 0:  # non empty
            masks = mmcv.concat_list(mask_results)
            inds = np.where(bboxes[:, -1] > score_thr)[0]
            np.random.seed(42)
            color_masks = [
                np.random.randint(0, 256, (1, 3), dtype=np.uint8)
                for _ in range(max(labels) + 1)
            ]
            for i in inds:
                i = int(i)
                color_mask = color_masks[labels[i]]
                mask = masks[i].astype(bool)
                img[mask] = img[mask] * 0.5 + color_mask * 0.5
        # if out_file specified, do not show image in window
        if out_file is not None:
            show = False
        # draw bounding boxes
        self.imshow_det_bboxes(
            img,
            bboxes,
            labels,
            class_names=self.CLASSES,
            score_thr=score_thr,
            thickness=thickness,
            font_scale=font_scale,
            win_name=win_name,
            show=show,
            wait_time=wait_time,
            out_file=out_file)

        if not (show or out_file):
            return img

    def imshow_det_bboxes(self,
                        img: Union[str, np.ndarray],
                        bboxes: np.ndarray,
                        labels: np.ndarray,
                        class_names: List[str] = None,
                        score_thr: float = 0,
                        thickness: int = 1,
                        font_scale: float = 0.5,
                        show: bool = True,
                        win_name: str = '',
                        wait_time: int = 0,
                        out_file: Optional[str] = None):
        """Draw bboxes and class labels (with scores) on an image.

        Args:
            img (str or ndarray): The image to be displayed.
            bboxes (ndarray): Bounding boxes (with scores), shaped (n, 4) or
                (n, 5).
            labels (ndarray): Labels of bboxes.
            class_names (list[str]): Names of each classes.
            score_thr (float): Minimum score of bboxes to be shown.
            bbox_color (Color or str or tuple or int or ndarray): Color
                of bbox lines.
            text_color (Color or str or tuple or int or ndarray): Color
                of texts.
            thickness (int): Thickness of lines.
            font_scale (float): Font scales of texts.
            show (bool): Whether to show the image.
            win_name (str): The window name.
            wait_time (int): Value of waitKey param.
            out_file (str or None): The filename to write the image.

        Returns:
            ndarray: The image with bboxes drawn on it.
        """
        assert bboxes.ndim == 2
        assert labels.ndim == 1
        assert bboxes.shape[0] == labels.shape[0]
        assert bboxes.shape[1] == 4 or bboxes.shape[1] == 5
        img = imread(img)
        img = np.ascontiguousarray(img)
        colors = [(3, 1, 210), (228, 8, 10), (254, 153, 0), (125, 218, 88),
                  (239, 195, 202), (255, 222, 89), (191, 214, 65),
                  (93, 226, 231), (6, 2, 112), (204, 108, 231)]
        
        if score_thr > 0:
            assert bboxes.shape[1] == 5
            scores = bboxes[:, -1]
            inds = np.argsort(scores)[::-1]
            bboxes = bboxes[inds]
            labels = labels[inds]
            scores = scores[inds]
            
            inds = scores > score_thr
            labels, inds = np.unique(labels[inds], return_index=True)
            bboxes = bboxes[inds, :]
            colors = [colors[i] for i in labels]

        for bbox, label, color in zip(bboxes, labels, colors):
            bbox_int = bbox.astype(np.int32)
            left_top = (bbox_int[0], bbox_int[1])
            right_bottom = (bbox_int[2], bbox_int[3])
            cv2.rectangle(
                img, left_top, right_bottom, color, thickness=thickness)
            label_text = class_names[
                label] if class_names is not None else f'cls {label}'
            if len(bbox) > 4:
                label_text += f'|{bbox[-1]:.02f}'
            cv2.putText(img, label_text, (bbox_int[0], bbox_int[1] - 2),
                        cv2.FONT_HERSHEY_COMPLEX, 1, (0,0,0), 2)

        if show:
            imshow(img, win_name, wait_time)
        if out_file is not None:
            imwrite(img, out_file)
        return img
    
def imshow(img: Union[str, np.ndarray],
           win_name: str = '',
           wait_time: int = 0):
    """Show an image.

    Args:
        img (str or ndarray): The image to be displayed.
        win_name (str): The window name.
        wait_time (int): Value of waitKey param.
    """
    cv2.imshow(win_name, imread(img))
    if wait_time == 0:  # prevent from hanging if windows was closed
        while True:
            ret = cv2.waitKey(1)

            closed = cv2.getWindowProperty(win_name, cv2.WND_PROP_VISIBLE) < 1
            # if user closed window or if some key pressed
            if closed or ret != -1:
                break
    else:
        ret = cv2.waitKey(wait_time)