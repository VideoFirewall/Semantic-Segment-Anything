import os
import torch
import torch.nn.functional as F
from PIL import Image
import mmcv
from tqdm import tqdm
from mmcv.utils import print_log
from mmdet.core.visualization.image import imshow_det_bboxes
from mmseg.core import intersect_and_union, pre_eval_to_metrics
from collections import OrderedDict
from prettytable import PrettyTable
import numpy as np
import pycocotools.mask as maskUtils
from configs.ade20k_id2label import CONFIG as CONFIG_ADE20K_ID2LABEL
from configs.coco_id2label import CONFIG as CONFIG_COCO_ID2LABEL
from clip import clip_classification
from clipseg import clipseg_segmentation
from oneformer import oneformer_coco_segmentation, oneformer_ade20k_segmentation, oneformer_cityscapes_segmentation
from blip import open_vocabulary_classification_blip
from segformer import segformer_segmentation as segformer_func

import cv2
import numpy as np
import torchvision.transforms as T

oneformer_func = {
    'ade20k': oneformer_ade20k_segmentation,
    'coco': oneformer_coco_segmentation,
    'cityscapes': oneformer_cityscapes_segmentation,
    'foggy_driving': oneformer_cityscapes_segmentation
}

def imshow_det_bboxes_with_masks(
    img, bboxes, labels, masks, class_names=None, colors=None, thickness=2, label_position="mask", show=True, out_file=None
):
    """
    Draw bounding boxes and masks on the image with labels at the specified position.

    Args:
        img (numpy.ndarray): The image to draw on.
        bboxes (list of list of int): Bounding boxes, each represented as [x1, y1, w, h].
        labels (list of int): Labels for each bounding box.
        masks (numpy.ndarray): Masks for each bounding box.
        class_names (list of str, optional): Class names for each label.
        colors (list of tuple of int, optional): Colors for each bounding box in BGR format.
        thickness (int, optional): Thickness of the bounding box lines.
        label_position (str, optional): Where to place the label ('bbox' or 'mask').
        show (bool, optional): Whether to display the image.
        out_file (str, optional): Path to save the image.
    """
    assert len(bboxes) == len(labels) == len(masks), "Lengths of bboxes, labels, and masks must match."

    if colors is None:
        # Generate random colors if not provided
        colors = [tuple(np.random.randint(0, 255, 3).tolist()) for _ in range(len(bboxes))]

    for bbox, label, mask, color in zip(bboxes, labels, masks, colors):
        x1, y1, w, h = bbox
        x2 = x1 + w
        y2 = y1 + h

        # Draw bounding box
        # cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)

        # Calculate label position
        if label_position == "bbox":
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
        elif label_position == "mask":
            mask_indices = np.argwhere(mask)
            if mask_indices.size > 0:
                center_y, center_x = mask_indices.mean(axis=0).astype(int)
            else:
                # Fallback to bounding box center if mask is empty
                center_x = (x1 + x2) // 2
                center_y = (y1 + y2) // 2
        else:
            raise ValueError("label_position must be 'bbox' or 'mask'.")

        # Draw label
        if class_names:
            label_text = class_names[label]
            (text_width, text_height), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
            text_x = center_x - text_width // 2
            text_y = center_y + text_height // 2
            cv2.rectangle(
                img,
                (text_x, text_y - text_height - baseline),
                (text_x + text_width, text_y + baseline),
                (0, 0, 0),
                thickness=cv2.FILLED,
            )
            cv2.putText(img, label_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        # Apply mask with alpha blending
        mask = mask.astype(bool)
        alpha = 0.5  # Transparency level for masks
        img[mask] = (1 - alpha) * img[mask] + alpha * np.array(color)

    # Show or save the result
    if show:
        cv2.imshow('Detections', img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()

    if out_file:
        cv2.imwrite(out_file, img)


def load_filename_with_extensions(data_path, filename):
    """
    Returns file with corresponding extension to json file.
    Raise error if such file is not found.

    Args:
        filename (str): Filename (without extension).

    Returns:
        filename with the right extension.
    """
    full_file_path = os.path.join(data_path, filename)
    # List of image file extensions to attempt
    image_extensions = ['.png', '.jpg', '.jpeg', '.gif', '.bmp', '.tiff']
    # Iterate through image file extensions and attempt to upload the file
    for ext in image_extensions:
        # Check if the file with current extension exists
        if os.path.exists(full_file_path + ext):
            return full_file_path + ext  # Return True if file is successfully uploaded
    raise FileNotFoundError(f"No such file {full_file_path}, checked for the following extensions {image_extensions}")

def semantic_annotation_pipeline(filename, data_path, output_path, rank, save_img=False, scale_small=1.2, scale_large=1.6, scale_huge=1.6,
                                 clip_processor=None,
                                 clip_model=None,
                                 oneformer_ade20k_processor=None,
                                 oneformer_ade20k_model=None,
                                 oneformer_coco_processor=None,
                                 oneformer_coco_model=None,
                                 blip_processor=None,
                                 blip_model=None,
                                 clipseg_processor=None,
                                 clipseg_model=None,
                                 mask_generator=None):
    img = mmcv.imread(load_filename_with_extensions(data_path, filename))
    if mask_generator is None:
        anns = mmcv.load(os.path.join(data_path, filename+'.json'))
    else:
        # start = time.time()
        # for _ in range(10):
        anns = {'annotations': mask_generator.generate(img)}
        # runtime = (time.time() - start) / 10
        # print(f"SAM Runtime of {data_path} image having {len(anns['annotations'])}: {runtime:.2f} seconds")
    bitmasks, class_names, bboxes = [], [], []
    
    # start = time.time()
    # for _ in range(10):
    class_ids_from_oneformer_coco = oneformer_coco_segmentation(Image.fromarray(img),oneformer_coco_processor,oneformer_coco_model, rank)
    class_ids_from_oneformer_ade20k = oneformer_ade20k_segmentation(Image.fromarray(img),oneformer_ade20k_processor,oneformer_ade20k_model, rank)
    # runtime = (time.time() - start) / 10
    # print(f"Semantic Seg Runtime of {data_path} image: {runtime:.2f} seconds")

    # start = time.time()
    # for _ in range(10):
    for ann in anns['annotations']:
        valid_mask = torch.tensor(maskUtils.decode(ann['segmentation'])).bool()
        # get the class ids of the valid pixels
        coco_propose_classes_ids = class_ids_from_oneformer_coco[valid_mask]
        ade20k_propose_classes_ids = class_ids_from_oneformer_ade20k[valid_mask]
        top_k_coco_propose_classes_ids = torch.bincount(coco_propose_classes_ids.flatten()).topk(1).indices
        top_k_ade20k_propose_classes_ids = torch.bincount(ade20k_propose_classes_ids.flatten()).topk(1).indices
        local_class_names = set()
        local_class_names = set.union(local_class_names, set([CONFIG_ADE20K_ID2LABEL['id2label'][str(class_id.item())] for class_id in top_k_ade20k_propose_classes_ids]))
        local_class_names = set.union(local_class_names, set(([CONFIG_COCO_ID2LABEL['refined_id2label'][str(class_id.item())] for class_id in top_k_coco_propose_classes_ids])))
        patch_small = mmcv.imcrop(img, np.array(
            [ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3]]),
                                  scale=scale_small)
        patch_large = mmcv.imcrop(img, np.array(
            [ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3]]),
                                  scale=scale_large)
        patch_huge = mmcv.imcrop(img, np.array(
            [ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3]]),
                                  scale=scale_huge)
        valid_mask_huge_crop = mmcv.imcrop(valid_mask.numpy(), np.array(
            [ann['bbox'][0], ann['bbox'][1], ann['bbox'][0] + ann['bbox'][2], ann['bbox'][1] + ann['bbox'][3]]),
                                    scale=scale_huge)
        op_class_list = open_vocabulary_classification_blip(patch_large,blip_processor, blip_model, rank)
        local_class_list = list(set.union(local_class_names, set(op_class_list))) # , set(refined_imagenet_class_names)
        mask_categories = clip_classification(patch_small, local_class_list, 3 if len(local_class_list)> 3 else len(local_class_list), clip_processor, clip_model, rank)
        class_ids_patch_huge = clipseg_segmentation(patch_huge, mask_categories, clipseg_processor, clipseg_model, rank).argmax(0)
        valid_mask_huge_crop = torch.tensor(valid_mask_huge_crop)
        if valid_mask_huge_crop.shape != class_ids_patch_huge.shape:
            valid_mask_huge_crop = F.interpolate(
                valid_mask_huge_crop.unsqueeze(0).unsqueeze(0).float(),
                size=(class_ids_patch_huge.shape[-2], class_ids_patch_huge.shape[-1]),
                mode='nearest').squeeze(0).squeeze(0).bool()
        top_1_patch_huge = torch.bincount(class_ids_patch_huge[valid_mask_huge_crop].flatten()).topk(1).indices
        top_1_mask_category = mask_categories[top_1_patch_huge.item()]

        ann['class_name'] = str(top_1_mask_category)
        ann['class_proposals'] = mask_categories
        class_names.append(str(top_1_mask_category))
        bboxes.append(ann['bbox'])
        # bitmasks.append(maskUtils.decode(ann['segmentation']))

        # Delete variables that are no longer needed
        del coco_propose_classes_ids
        del ade20k_propose_classes_ids
        del top_k_coco_propose_classes_ids
        del top_k_ade20k_propose_classes_ids
        del patch_small
        del patch_large
        del patch_huge
        del valid_mask_huge_crop
        del op_class_list
        del mask_categories
        del class_ids_patch_huge
    # runtime = (time.time() - start) / 10
    # print(f"Semantic Voting Runtime of {data_path} image: {runtime:.2f} seconds")
    mmcv.dump(anns, os.path.join(output_path, filename + '_semantic.json'))
    print('[Save] save SSA-engine annotation results: ', os.path.join(output_path, filename + '_semantic.json'))
    if save_img:
        for ann in anns['annotations']:
            bitmasks.append(maskUtils.decode(ann['segmentation']))
        imshow_det_bboxes(img,
                    bboxes=None,
                    labels=np.arange(len(bitmasks)),
                    segms=np.stack(bitmasks),
                    class_names=class_names,
                    font_size=25,
                    show=False,
                    out_file=os.path.join(output_path, filename+'_semantic.png'))

    # Delete variables that are no longer needed
    del img
    del anns
    del class_ids_from_oneformer_coco
    del class_ids_from_oneformer_ade20k

def img_load(data_path, filename, dataset):
    # load image
    if dataset == 'ade20k':
        img = mmcv.imread(os.path.join(data_path, filename+'.jpg'))
    elif dataset == 'cityscapes' or dataset == 'foggy_driving':
        img = mmcv.imread(os.path.join(data_path, filename+'.png'))
    else:
        raise NotImplementedError()
    return img

def generate_tile(clean_tensor: torch.Tensor, UAP_tensor: torch.Tensor):
    adv_height, adv_width = UAP_tensor.size(1), UAP_tensor.size(2)
    clean_height, clean_width = clean_tensor.size(1), clean_tensor.size(2)
    num_copies_horizontally = int(np.ceil(clean_width / adv_width))
    num_copies_vertically = int(np.ceil(clean_height / adv_height))
    adv_patch_h = UAP_tensor
    for i in range(num_copies_horizontally - 1):
        adv_patch_h = torch.cat((adv_patch_h, UAP_tensor), dim=2)
    adv_patch_v = adv_patch_h
    for i in range(num_copies_vertically - 1):
        adv_patch_v = torch.cat((adv_patch_v, adv_patch_h), dim=1)
    tiled_UAP_tensor = adv_patch_v[:, :clean_height, :clean_width]

    return tiled_UAP_tensor

def perturb(image: np.array, uap_path: str, uap_method: str = 'Upscaled') -> np.array:
        img_tensor = T.ToTensor()(Image.fromarray(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))) 
        _, height, width = img_tensor.shape
        
        UAP_img = mmcv.imread(uap_path)
        UAP_arr = Image.fromarray(cv2.cvtColor(UAP_img, cv2.COLOR_BGR2RGB))
        if uap_method == 'Upscaled':
            preprocess = T.Compose([
                T.Resize((height, width)),
                T.ToTensor(),
            ])
            UAP_tensor = preprocess(UAP_arr)
        elif uap_method == 'Tiled':
            UAP_tensor = T.ToTensor()(UAP_arr)
            UAP_tensor = generate_tile(img_tensor, UAP_tensor)
        else:
            raise NotImplementedError()

        perturbed_tensor = torch.clamp(img_tensor + UAP_tensor, 0, 1)
        perturbed_image = perturbed_tensor.permute(1, 2, 0).mul(255).byte().numpy() 

        return cv2.cvtColor(perturbed_image, cv2.COLOR_RGB2BGR) 

def semantic_segment_anything_inference(filename, output_path, rank, img=None, save_img=False,
                                 semantic_branch_processor=None,
                                 semantic_branch_model=None,
                                 mask_branch_model=None,
                                 dataset=None,
                                 id2label=None,
                                 model='segformer'):

    anns = {'annotations': mask_branch_model.generate(img)}
    h, w, _ = img.shape
    bitmasks, class_names = [], []
    if model == 'oneformer':
        class_ids = oneformer_func[dataset](Image.fromarray(img), semantic_branch_processor,
                                                                        semantic_branch_model, rank)
    elif model == 'segformer':
        class_ids = segformer_func(img, semantic_branch_processor, semantic_branch_model, rank)
    else:
        raise NotImplementedError()
    semantc_mask = class_ids.clone()
    anns['annotations'] = sorted(anns['annotations'], key=lambda x: x['area'], reverse=True)
    for ann in anns['annotations']:
        valid_mask = torch.tensor(maskUtils.decode(ann['segmentation'])).bool()
        # get the class ids of the valid pixels
        propose_classes_ids = class_ids[valid_mask]
        num_class_proposals = len(torch.unique(propose_classes_ids))
        if num_class_proposals == 1:
            semantc_mask[valid_mask] = propose_classes_ids[0]
            ann['class_name'] = id2label['id2label'][str(propose_classes_ids[0].item())]
            ann['class_proposals'] = id2label['id2label'][str(propose_classes_ids[0].item())]
            class_names.append(ann['class_name'])
            # bitmasks.append(maskUtils.decode(ann['segmentation']))
            continue
        top_1_propose_class_ids = torch.bincount(propose_classes_ids.flatten()).topk(1).indices
        top_1_propose_class_names = [id2label['id2label'][str(class_id.item())] for class_id in top_1_propose_class_ids]

        semantc_mask[valid_mask] = top_1_propose_class_ids
        ann['class_name'] = top_1_propose_class_names[0]
        ann['class_proposals'] = top_1_propose_class_names[0]
        class_names.append(ann['class_name'])
        # bitmasks.append(maskUtils.decode(ann['segmentation']))

        del valid_mask
        del propose_classes_ids
        del num_class_proposals
        del top_1_propose_class_ids
        del top_1_propose_class_names
    
    sematic_class_in_img = torch.unique(semantc_mask)
    semantic_bitmasks, semantic_class_names = [], []

    # semantic prediction
    anns['semantic_mask'] = {}
    for i in range(len(sematic_class_in_img)):
        class_name = id2label['id2label'][str(sematic_class_in_img[i].item())]
        class_mask = semantc_mask == sematic_class_in_img[i]
        class_mask = class_mask.cpu().numpy().astype(np.uint8)
        semantic_class_names.append(class_name)
        semantic_bitmasks.append(class_mask)
        anns['semantic_mask'][str(sematic_class_in_img[i].item())] = maskUtils.encode(np.array((semantc_mask == sematic_class_in_img[i]).cpu().numpy(), order='F', dtype=np.uint8))
        anns['semantic_mask'][str(sematic_class_in_img[i].item())]['counts'] = anns['semantic_mask'][str(sematic_class_in_img[i].item())]['counts'].decode('utf-8')
    
    if save_img:
        for ann in anns['annotations']:
            bitmasks.append(maskUtils.decode(ann['segmentation']))
        imshow_det_bboxes(img,
                            bboxes=None,
                            labels=np.arange(len(bitmasks)),
                            segms=np.stack(bitmasks),
                            class_names=class_names,
                            font_size=25,
                            show=False,
                            out_file=os.path.join(output_path, filename + '_semantic.png'))
        print('[Save] save SSA prediction: ', os.path.join(output_path, filename + '_semantic.png'))
    mmcv.dump(anns, os.path.join(output_path, filename + '_semantic.json'))
    # 手动清理不再需要的变量
    del img
    del anns
    del class_ids
    del semantc_mask
    # del bitmasks
    del class_names
    del semantic_bitmasks
    del semantic_class_names

    # gc.collect()
    
def eval_pipeline(gt_path, res_path, dataset):
    logger = None
    if dataset == 'cityscapes' or dataset == 'foggy_driving':
        class_names = ('road', 'sidewalk', 'building', 'wall', 'fence', 'pole', 'traffic light', 'traffic sign', 'vegetation', 'terrain', 'sky', 'person', 'rider', 'car', 'truck', 'bus', 'train', 'motorcycle', 'bicycle')
    elif dataset == 'ade20k':
        class_names = ('wall', 'building', 'sky', 'floor', 'tree', 'ceiling', 'road', 'bed ', 'windowpane', 'grass', 'cabinet', 'sidewalk', 'person', 'earth', 'door', 'table', 'mountain', 'plant', 'curtain', 'chair', 'car', 'water', 'painting', 'sofa', 'shelf', 'house', 'sea', 'mirror', 'rug', 'field', 'armchair', 'seat', 'fence', 'desk', 'rock', 'wardrobe', 'lamp', 'bathtub', 'railing', 'cushion', 'base', 'box', 'column', 'signboard', 'chest of drawers', 'counter', 'sand', 'sink', 'skyscraper', 'fireplace', 'refrigerator', 'grandstand', 'path', 'stairs', 'runway', 'case', 'pool table', 'pillow', 'screen door', 'stairway', 'river', 'bridge', 'bookcase', 'blind', 'coffee table', 'toilet', 'flower', 'book', 'hill', 'bench', 'countertop', 'stove', 'palm', 'kitchen island', 'computer', 'swivel chair', 'boat', 'bar', 'arcade machine', 'hovel', 'bus', 'towel', 'light', 'truck', 'tower', 'chandelier', 'awning', 'streetlight', 'booth', 'television receiver', 'airplane', 'dirt track', 'apparel', 'pole', 'land', 'bannister', 'escalator', 'ottoman', 'bottle', 'buffet', 'poster', 'stage', 'van', 'ship', 'fountain', 'conveyer belt', 'canopy', 'washer', 'plaything', 'swimming pool', 'stool', 'barrel', 'basket', 'waterfall', 'tent', 'bag', 'minibike', 'cradle', 'oven', 'ball', 'food', 'step', 'tank', 'trade name', 'microwave', 'pot', 'animal', 'bicycle', 'lake', 'dishwasher', 'screen', 'blanket', 'sculpture', 'hood', 'sconce', 'vase', 'traffic light', 'tray', 'ashcan', 'fan', 'pier', 'crt screen', 'plate', 'monitor', 'bulletin board', 'shower', 'radiator', 'glass', 'clock', 'flag')
    file_client = mmcv.FileClient(**{'backend': 'disk'})
    pre_eval_results = []
    if dataset == 'cityscapes':
        prefixs = ['frankfurt','lindau','munster']
    elif dataset == 'foggy_driving':
        prefixs = ['public', 'pedestrian']
    elif dataset == 'ade20k':
        prefixs = ['']
    else:
        raise NotImplementedError
    for split in tqdm(prefixs, desc="Split loop"):
        gt_path_split = os.path.join(gt_path, split)
        res_path_split = os.path.join(res_path, split)
        filenames = [fn_ for fn_ in os.listdir(res_path_split) if '.json' in fn_]
        for i, fn_ in enumerate(tqdm(filenames, desc="File loop")):
            pred_fn = os.path.join(res_path_split, fn_)
            result = mmcv.load(pred_fn)
            num_classes = len(class_names)
            init_flag = True
            for id_str, mask in result['semantic_mask'].items():
                mask_ = maskUtils.decode(mask)
                h, w = mask_.shape
                if init_flag:
                    seg_mask = torch.zeros((1, 1, h, w))
                    init_flag = False
                mask_ = torch.from_numpy(mask_).unsqueeze(0).unsqueeze(0)
                seg_mask[mask_] = int(id_str)
            seg_logit = torch.zeros((1, num_classes, h, w))
            seg_logit.scatter_(1, seg_mask.long(), 1)
            seg_logit = seg_logit.float()
            seg_pred = F.softmax(seg_logit, dim=1).argmax(dim=1).squeeze(0).numpy()
            if dataset == 'cityscapes' or dataset == 'foggy_driving':
                gt_fn_ = os.path.join(gt_path_split, fn_.replace('_leftImg8bit_semantic.json','_gtFine_labelTrainIds.png'))
            elif dataset == 'ade20k':
                gt_fn_ = os.path.join(gt_path, fn_.replace('_semantic.json','.png'))
            img_bytes = file_client.get(gt_fn_)
            seg_map = mmcv.imfrombytes(
                img_bytes, flag='unchanged',
                backend='pillow').squeeze().astype(np.uint8)
            if dataset=='ade20k':
                seg_map = seg_map - 1
            pre_eval_results.append(intersect_and_union(
                            seg_pred,
                            seg_map,
                            num_classes,
                            255,
                            label_map=dict(),
                            reduce_zero_label=False))

    ret_metrics = pre_eval_to_metrics(pre_eval_results, ['mIoU'])
    ret_metrics_summary = OrderedDict({
                ret_metric: np.round(np.nanmean(ret_metric_value) * 100, 2)
                for ret_metric, ret_metric_value in ret_metrics.items()
            })
    # each class table
    ret_metrics.pop('aAcc', None)
    ret_metrics_class = OrderedDict({
        ret_metric: np.round(ret_metric_value * 100, 2)
        for ret_metric, ret_metric_value in ret_metrics.items()
    })
    ret_metrics_class.update({'Class': class_names})
    ret_metrics_class.move_to_end('Class', last=False)

    # for logger
    class_table_data = PrettyTable()
    for key, val in ret_metrics_class.items():
        class_table_data.add_column(key, val)

    summary_table_data = PrettyTable()
    for key, val in ret_metrics_summary.items():
        if key == 'aAcc':
            summary_table_data.add_column(key, [val])
        else:
            summary_table_data.add_column('m' + key, [val])

    print_log('per class results:', logger)
    print_log('\n' + class_table_data.get_string(), logger=logger)
    print_log('Summary:', logger)
    print_log('\n' + summary_table_data.get_string(), logger=logger)
