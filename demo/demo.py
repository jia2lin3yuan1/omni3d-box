# Copyright (c) Meta Platforms, Inc. and affiliates
import logging
import os
import argparse
import sys
import numpy as np
from collections import OrderedDict
import torch

from detectron2.checkpoint import DetectionCheckpointer
from detectron2.config import get_cfg
from detectron2.engine import default_argument_parser, default_setup, launch
from detectron2.data import transforms as T

logger = logging.getLogger("detectron2")

sys.dont_write_bytecode = True
sys.path.append(os.getcwd())
np.set_printoptions(suppress=True)

from cubercnn.config import get_cfg_defaults
from cubercnn.modeling.proposal_generator import RPNWithIgnore
from cubercnn.modeling.roi_heads import ROIHeads3D
from cubercnn.modeling.meta_arch import RCNN3D, build_model
from cubercnn.modeling.backbone import build_dla_from_vision_fpn_backbone
from cubercnn import util, vis


def fast_nms(scores, obj_bboxes, score_threshold=None, nms_threshold=0.7):
    '''
    @Param: scores -- tensor or ndarray in form [num_obj]
            obj_bboxes -- tensor or ndarray in form [num_obj, 4], with (x0, y0, x1, y1)
    @Output: ndarray, index of keep objects sorted with scores
    '''
    if isinstance(scores, np.ndarray):
        scores = torch.tensor(scores)
    if isinstance(obj_bboxes, np.ndarray):
        obj_bboxes = torch.tensor(obj_bboxes)

    # sorted masks
    sorted_scores, sorted_scores_indexes = torch.sort(scores, descending=True)
    sorted_bboxes = obj_bboxes[sorted_scores_indexes, :]
    x0, y0        = sorted_bboxes[:, 0], sorted_bboxes[:,1]
    x1, y1        = sorted_bboxes[:, 2], sorted_bboxes[:,3]
    areas         = (y1 - y0 + 1) * (x1 - x0 + 1)

    #Triangulation on iou matrix
    max_x0 = torch.maximum(x0[:, None], x0[None, :]) # [N, N]
    max_y0 = torch.maximum(y0[:, None], y0[None, :]) # [N, N]
    min_x1 = torch.minimum(x1[:, None], x1[None, :]) # [N, N]
    min_y1 = torch.minimum(y1[:, None], y1[None, :]) # [N, N]

    intp_x = torch.clip(min_x1 - max_x0 + 1, min=0)
    intp_y = torch.clip(min_y1 - max_y0 + 1, min=0)
    ovlp   = intp_x * intp_y
    union  = areas[:, None] + areas[None, :] - ovlp

    ious = ovlp / (union + 1.)
    ious = torch.triu(ious, diagonal=1)
    ious = ious.max(dim=0)[0]

    # NMS keep
    if score_threshold is not None:
        keep = (ious < nms_threshold) * (sorted_scores >= score_threshold)
    else:
        keep = ious < nms_threshold
    return sorted_scores_indexes[keep].detach().numpy()


def remove_duplicate_nms(K, meshes, scores, nms_thr=0.5, im_size=(480, 760)):
    verts3D = [ele.verts_padded()[0].numpy() for ele in meshes]
    verts2D = [(K @ ele.T) / ele[:, -1] for ele in verts3D]

    ht, wd = im_size
    x0 = np.clip(np.asarray([ele[0, :].min() for ele in verts2D]), 0, wd -1)
    y0 = np.clip(np.asarray([ele[1, :].min() for ele in verts2D]), 0, ht -1)
    x1 = np.clip(np.asarray([ele[0, :].max() for ele in verts2D]), 0, wd -1)
    y1 = np.clip(np.asarray([ele[1, :].max() for ele in verts2D]), 0, ht -1)
    bboxes_2d = np.vstack([x0, y0, x1,y1]).transpose(1,0)

    bh, bw = y1 - y0, x1 - x0
    keep = np.logical_and(bh < ht // 2, bw < wd // 2)
    if keep.sum() == 0:
        keep[0] = True

    scores = scores[keep]
    bboxes_2d = bboxes_2d[keep]
    keep = fast_nms(scores, bboxes_2d, nms_threshold=nms_thr)

    return keep


def do_test(args, cfg, model):

    list_of_ims = util.list_files(os.path.join(args.input_folder, ''), '*')

    model.eval()

    focal_length = args.focal_length
    principal_point = args.principal_point
    thres = args.threshold

    output_dir = cfg.OUTPUT_DIR
    min_size = cfg.INPUT.MIN_SIZE_TEST
    max_size = cfg.INPUT.MAX_SIZE_TEST
    augmentations = T.AugmentationList([T.ResizeShortestEdge(min_size, max_size, "choice")])

    util.mkdir_if_missing(output_dir)

    category_path = os.path.join(util.file_parts(args.config_file)[0], 'category_meta.json')

    # store locally if needed
    if category_path.startswith(util.CubeRCNNHandler.PREFIX):
        category_path = util.CubeRCNNHandler._get_local_path(util.CubeRCNNHandler, category_path)

    metadata = util.load_json(category_path)
    cats = metadata['thing_classes']

    for tk, path in enumerate(list_of_ims):
        if tk % 20 == 0:
            print(f'Processing {tk} | {len(list_of_ims)}')

        im_name = util.file_parts(path)[1]
        im = util.imread(path)

        if im is None:
            continue

        image_shape = im.shape[:2]  # h, w

        h, w = image_shape

        if focal_length == 0:
            focal_length_ndc = 4.0
            focal_length = focal_length_ndc * h / 2

        if len(principal_point) == 0:
            px, py = w/2, h/2
        else:
            px, py = principal_point

        K = np.array([
            [focal_length, 0.0, px],
            [0.0, focal_length, py],
            [0.0, 0.0, 1.0]
        ])

        aug_input = T.AugInput(im)
        _ = augmentations(aug_input)
        image = aug_input.image

        batched = [{
            'image': torch.as_tensor(np.ascontiguousarray(image.transpose(2, 0, 1))).cuda(),
            'height': image_shape[0], 'width': image_shape[1], 'K': K
        }]

        dets = model(batched)[0]['instances']
        n_det = len(dets)

        meshes = []
        meshes_text = []
        #     if n_det > 0:
        #         dets.pred_bbox3D     = dets.pred_bbox3D[keep]
        #         dets.pred_center_cam = dets.pred_center_cam[keep]
        #         dets.pred_center_2D  = dets.pred_center_2D[keep]
        #         dets.pred_dimensions = dets.pred_dimensions[keep]
        #         dets.pred_pose       = dets.pred_pose[keep]
        #         dets.scores          = dets.scores[keep]
        #         dets.pred_classes    = dets.pred_classes[keep]

        if n_det > 0:
            for idx, (corners3D, center_cam, center_2D, dimensions, pose, score, cat_idx) in enumerate(zip(
                    dets.pred_bbox3D, dets.pred_center_cam, dets.pred_center_2D, dets.pred_dimensions,
                    dets.pred_pose, dets.scores, dets.pred_classes
                )):

                # skip
                if score < thres:
                    continue

                cat = cats[cat_idx]
                bbox3D = center_cam.tolist() + dimensions.tolist()
                meshes_text.append('{} {:.2f}'.format(cat, score))
                color = [c/255.0 for c in util.get_color(idx)]
                box_mesh = util.mesh_cuboid(bbox3D, pose.tolist(), color=color)
                meshes.append(box_mesh)

        if len(meshes) > 0:
            scores = np.asarray([float(ele.split(' ')[-1]) for ele in meshes_text])
            keep = remove_duplicate_nms(K, meshes, scores, nms_thr=0.1, im_size=im.shape[:2])
            meshes = [meshes[v] for v in keep]

        print('File: {} with {} dets'.format(im_name, len(meshes)))
        if len(meshes) > 0:
            im_drawn_rgb, im_topdown, _ = vis.draw_scene_view(im, K, meshes, text=meshes_text, scale=im.shape[0], blend_weight=0.5, blend_weight_overlay=0.85)

            if args.display:
                im_concat = np.concatenate((im_drawn_rgb, im_topdown), axis=1)
                vis.imshow(im_concat)

            util.imwrite(im_drawn_rgb, os.path.join(output_dir, im_name+'_boxes.jpg'))
            # util.imwrite(im_topdown, os.path.join(output_dir, im_name+'_novel.jpg'))
        else:
            util.imwrite(im, os.path.join(output_dir, im_name+'_boxes.jpg'))



def setup(args):
    """
    Create configs and perform basic setups.
    """
    cfg = get_cfg()
    get_cfg_defaults(cfg)

    config_file = args.config_file

    # store locally if needed
    if config_file.startswith(util.CubeRCNNHandler.PREFIX):
        config_file = util.CubeRCNNHandler._get_local_path(util.CubeRCNNHandler, config_file)

    cfg.merge_from_file(config_file)
    cfg.merge_from_list(args.opts)
    cfg.freeze()
    default_setup(cfg, args)
    return cfg

def main(args):
    cfg = setup(args)
    model = build_model(cfg)

    logger.info("Model:\n{}".format(model))
    DetectionCheckpointer(model, save_dir=cfg.OUTPUT_DIR).resume_or_load(
        cfg.MODEL.WEIGHTS, resume=True
    )

    with torch.no_grad():
        do_test(args, cfg, model)

if __name__ == "__main__":

    parser = argparse.ArgumentParser(
        epilog=None, formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--config-file", default="", metavar="FILE", help="path to config file")
    parser.add_argument('--input-folder',  type=str, help='list of image folders to process', required=True)
    parser.add_argument("--focal-length", type=float, default=0, help="focal length for image inputs (in px)")
    parser.add_argument("--principal-point", type=float, default=[], nargs=2, help="principal point for image inputs (in px)")
    parser.add_argument("--threshold", type=float, default=0.25, help="threshold on score for visualizing")
    parser.add_argument("--display", default=False, action="store_true", help="Whether to show the images in matplotlib",)

    parser.add_argument("--eval-only", default=True, action="store_true", help="perform evaluation only")
    parser.add_argument("--num-gpus", type=int, default=1, help="number of gpus *per machine*")
    parser.add_argument("--num-machines", type=int, default=1, help="total number of machines")
    parser.add_argument(
        "--machine-rank", type=int, default=0, help="the rank of this machine (unique per machine)"
    )
    port = 2 ** 15 + 2 ** 14 + hash(os.getuid() if sys.platform != "win32" else 1) % 2 ** 14
    parser.add_argument(
        "--dist-url",
        default="tcp://127.0.0.1:{}".format(port),
        help="initialization URL for pytorch distributed backend. See "
        "https://pytorch.org/docs/stable/distributed.html for details.",
    )
    parser.add_argument(
        "opts",
        help="Modify config options by adding 'KEY VALUE' pairs at the end of the command. "
        "See config references at "
        "https://detectron2.readthedocs.io/modules/config.html#config-references",
        default=None,
        nargs=argparse.REMAINDER,
    )

    args = parser.parse_args()

    print("Command Line Args:", args)
    launch(
        main,
        args.num_gpus,
        num_machines=args.num_machines,
        machine_rank=args.machine_rank,
        dist_url=args.dist_url,
        args=(args,),
    )
