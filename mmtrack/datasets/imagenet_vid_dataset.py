# Copyright (c) OpenMMLab. All rights reserved.
from mmdet.datasets import DATASETS
from mmdet.datasets.api_wrappers import COCO

from .coco_video_dataset import CocoVideoDataset
from .parsers import CocoVID


@DATASETS.register_module()
class ImagenetVIDDataset(CocoVideoDataset):
    """ImageNet VID dataset for video object detection."""

    CLASSES = ('airplane', 'antelope', 'bear', 'bicycle', 'bird', 'bus', 'car',
               'cattle', 'dog', 'domestic_cat', 'elephant', 'fox',
               'giant_panda', 'hamster', 'horse', 'lion', 'lizard', 'monkey',
               'motorcycle', 'rabbit', 'red_panda', 'sheep', 'snake',
               'squirrel', 'tiger', 'train', 'turtle', 'watercraft', 'whale',
               'zebra')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_annotations(self, ann_file):
        """Load annotations from COCO/COCOVID style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation information from COCO/COCOVID api.
        """
        if self.load_as_video:
            data_infos = self.load_video_anns(ann_file)
        else:
            data_infos = self.load_image_anns(ann_file)
        return data_infos

    def load_image_anns(self, ann_file):
        """Load annotations from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation information from COCO api.
        """
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}

        all_img_ids = self.coco.get_img_ids()
        self.img_ids = []
        data_infos = []
        for img_id in all_img_ids:
            info = self.coco.load_imgs([img_id])[0]
            info['filename'] = info['file_name']
            if info['is_vid_train_frame']:
                self.img_ids.append(img_id)
                data_infos.append(info)
        return data_infos

    def load_video_anns(self, ann_file):
        """Load annotations from COCOVID style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation information from COCOVID api.
        """
        self.coco = CocoVID(ann_file)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}

        data_infos = []
        self.vid_ids = self.coco.get_vid_ids()
        self.img_ids = []
        for vid_id in self.vid_ids:
            img_ids = self.coco.get_img_ids_from_vid(vid_id)
            for img_id in img_ids:
                info = self.coco.load_imgs([img_id])[0]
                info['filename'] = info['file_name']
                if self.test_mode:
                    assert not info['is_vid_train_frame'], \
                        'is_vid_train_frame must be False in testing'
                    self.img_ids.append(img_id)
                    data_infos.append(info)
                elif info['is_vid_train_frame']:
                    self.img_ids.append(img_id)
                    data_infos.append(info)
        return data_infos


@DATASETS.register_module()
class SUITDataset(CocoVideoDataset):
    """ImageNet VID dataset for video object detection."""

    CLASSES = ('C5', 'C6', 'C7', 'C8', 'UT', 'MT', 'LT', 'SSN', 'AD', 'PD')

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def load_annotations(self, ann_file):
        """Load annotations from COCO/COCOVID style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation information from COCO/COCOVID api.
        """
        if self.load_as_video:
            data_infos = self.load_video_anns(ann_file)
        else:
            data_infos = self.load_image_anns(ann_file)
        return data_infos

    def load_image_anns(self, ann_file):
        """Load annotations from COCO style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation information from COCO api.
        """
        self.coco = COCO(ann_file)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}

        all_img_ids = self.coco.get_img_ids()
        self.img_ids = []
        data_infos = []
        for img_id in all_img_ids:
            info = self.coco.load_imgs([img_id])[0]
            info['filename'] = info['file_name']
            # if info['is_vid_train_frame']:
            self.img_ids.append(img_id)
            data_infos.append(info)
        return data_infos

    def load_video_anns(self, ann_file):
        """Load annotations from COCOVID style annotation file.

        Args:
            ann_file (str): Path of annotation file.

        Returns:
            list[dict]: Annotation information from COCOVID api.
        """
        self.coco = CocoVID(ann_file)
        self.cat_ids = self.coco.get_cat_ids(cat_names=self.CLASSES)
        self.cat2label = {cat_id: i for i, cat_id in enumerate(self.cat_ids)}

        data_infos = []
        self.vid_ids = self.coco.get_vid_ids()
        self.img_ids = []
        for vid_id in self.vid_ids:
            img_ids = self.coco.get_img_ids_from_vid(vid_id)
            for img_id in img_ids:
                info = self.coco.load_imgs([img_id])[0]
                info['filename'] = info['file_name']
                # if self.test_mode:
                #     assert not info['is_vid_train_frame'], \
                #         'is_vid_train_frame must be False in testing'
                #     self.img_ids.append(img_id)
                #     data_infos.append(info)
                # elif info['is_vid_train_frame']:
                self.img_ids.append(img_id)
                data_infos.append(info)
        return data_infos

    def ref_img_sampling(self,
                         img_info,
                         ref_indices):
        """Sampling reference frames in the same video for key frame.

        Args:
            img_info (dict): The information of key frame.
            frame_range (List(int) | int): The sampling range of reference
                frames in the same video for key frame.
            stride (int): The sampling frame stride when sampling reference
                images. Default: 1.
            num_ref_imgs (int): The number of sampled reference images.
                Default: 1.
            filter_key_img (bool): If False, the key image will be in the
                sampling reference candidates, otherwise, it is exclude.
                Default: True.
            method (str): The sampling method. Options are 'uniform',
                'bilateral_uniform', 'test_with_adaptive_stride',
                'test_with_fix_stride'. 'uniform' denotes reference images are
                randomly sampled from the nearby frames of key frame.
                'bilateral_uniform' denotes reference images are randomly
                sampled from the two sides of the nearby frames of key frame.
                'test_with_adaptive_stride' is only used in testing, and
                denotes the sampling frame stride is equal to (video length /
                the number of reference images). test_with_fix_stride is only
                used in testing with sampling frame stride equalling to
                `stride`. Default: 'uniform'.
            return_key_img (bool): If True, the information of key frame is
                returned, otherwise, not returned. Default: True.

        Returns:
            list(dict): `img_info` and the reference images information or
            only the reference images information.
        """
        assert isinstance(img_info, dict)
        assert isinstance(ref_indices, list)
        assert len(ref_indices) > 0
        for i in ref_indices:
            assert isinstance(i, int), 'Each element must be int.'

        if (not self.load_as_video) or img_info.get('frame_id', -1) < 0:
            ref_img_infos = []
            for i in range(len(ref_indices)):
                ref_img_infos.append(img_info.copy())
        else:
            vid_id, img_id, frame_id = img_info['video_id'], img_info[
                'id'], img_info['frame_id']
            img_ids = self.coco.get_img_ids_from_vid(vid_id)
            ref_img_ids = []
            for i in range(len(ref_indices)):  
                if 0 <= frame_id + i and frame_id + i < len(img_ids):
                    ref_img_ids.append(img_ids[frame_id + i])

            ref_img_infos = []
            for ref_img_id in ref_img_ids:
                ref_img_info = self.coco.load_imgs([ref_img_id])[0]
                ref_img_info['filename'] = ref_img_info['file_name']
                ref_img_infos.append(ref_img_info)
            ref_img_infos = sorted(ref_img_infos, key=lambda i: i['frame_id'])

        return [img_info, *ref_img_infos]