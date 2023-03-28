import argparse
import glob
import os
import gc

import cv2
import mmcv
import numpy as np
from tqdm import tqdm
import torch
from mmcv.runner import load_checkpoint
from mmedit.core import tensor2img
import collections
from concurrent.futures import ThreadPoolExecutor
import time
from collections import deque
from utils import VideoReader, VideoWriter

from realbasicvsr.models.builder import build_model

VIDEO_EXTENSIONS = ('mp4', 'mov')

class RestoreImageWrapper:
    def __init__(self, restored_img=None, is_ended=False):
        self.restored_img = restored_img
        self.is_ended = is_ended


def parse_args():
    parser = argparse.ArgumentParser(
        description='Inference script of RealBasicVSR')
    parser.add_argument('--config', type=str, default='configs/realbasicvsr_x4.py', help='test config file path')
    parser.add_argument('--checkpoint', type=str, default='checkpoints/RealBasicVSR_x4.pth', help='checkpoint file')
    parser.add_argument('input_dir', help='directory of the input video')
    parser.add_argument('--output_dir', type=str, default="results", help='directory of the output video')
    parser.add_argument(
        '--max_seq_len',
        type=int,
        default=None,
        help='maximum sequence length to be processed')
    parser.add_argument(
        '--is_save_as_png',
        type=bool,
        default=True,
        help='whether to save as png')
    parser.add_argument(
        '--fps', type=float, default=25, help='FPS of the output video')
    args = parser.parse_args()

    return args


def init_model(config, checkpoint=None):
    """Initialize a model from config file.

    Args:
        config (str or :obj:`mmcv.Config`): Config file path or the config
            object.
        checkpoint (str, optional): Checkpoint path. If left as None, the model
            will not load any weights.
        device (str): Which device the model will deploy. Default: 'cuda:0'.

    Returns:
        nn.Module: The constructed model.
    """

    if isinstance(config, str):
        config = mmcv.Config.fromfile(config)
    elif not isinstance(config, mmcv.Config):
        raise TypeError('config must be a filename or Config object, '
                        f'but got {type(config)}')
    config.model.pretrained = None
    config.test_cfg.metrics = None
    model = build_model(config.model, test_cfg=config.test_cfg)
    if checkpoint is not None:
        checkpoint = load_checkpoint(model, checkpoint)

    model.cfg = config  # save the config in the model for convenience
    model.eval()

    return model


def save_as_video_async(result_root, video_name, fps, audio, restore_img_dqueue):
    print('Video Saving...')
    if not os.path.exists(result_root):
        os.makedirs(result_root)

    video_full_name = f'{video_name}.mp4'
    save_restore_path = os.path.join(result_root, video_full_name)

    video_writer = None
    while True:
        if len(restore_img_dqueue) <= 0:
            time.sleep(1)
            continue
        restore_img_wrapper = restore_img_dqueue.popleft()
        if restore_img_wrapper.is_ended:
            break

        if video_writer is None:
            height, width = restore_img_wrapper.restored_img.shape[:2]
            video_writer = VideoWriter(save_restore_path, height, width, fps, audio)

        try:
            video_writer.write_frame(restore_img_wrapper.restored_img)
        except Exception as e:
            print(f"error:{e}")

    if video_writer is not None:
        video_writer.close()


if __name__ == '__main__':
    args = parse_args()

    start_time = time.time()
    restore_img_dqueue = collections.deque()
    # torch.backends.cudnn.benchmark = True

    # initialize the model
    model = init_model(args.config, args.checkpoint)
    video_reader = None
    # read images
    full_video_name = os.path.basename(args.input_dir)
    video_name, file_extension = full_video_name.split('.')
    if file_extension in VIDEO_EXTENSIONS:  # input is a video file
        video_reader = VideoReader(args.input_dir)
    else:
        raise ValueError('"input_dir" can only be a video or a directory.')

    fps = video_reader.get_fps()
    audio = video_reader.audio
    thread_pool = ThreadPoolExecutor()
    video_save_feature = thread_pool.submit(save_as_video_async, args.output_dir, video_name,
                                            fps, audio, restore_img_dqueue)

    # map to cuda, if available
    cuda_flag = False
    if torch.cuda.is_available():
        model = model.cuda()
        cuda_flag = True

    total_count = video_reader.nb_frames
    step = args.max_seq_len
    if step is None:
        coe = 2
        minWH = min(video_reader.height, video_reader.width)
        if minWH <= 320:
            coe = 3
        elif minWH <= 480:
            coe = 2
        elif minWH <= 720:
            coe = 1
        elif minWH <= 1080:
            coe = 0.2
        else:
            coe = 0.001
        step = max(1, int(25 * coe))
        print(f'\nminWH:{minWH}, step:{step}')
    for i in tqdm(range(0, total_count, step)):
        torch.cuda.empty_cache()
        start = i
        end = i + step
        if end > total_count:
            end = total_count

        frame_group = []
        for j in range(start, end):
            try:
                frame = video_reader.get_frame()
                frame = np.flip(frame, axis=2)
                if frame is None:
                    continue
                frame = torch.from_numpy(frame / 255.).permute(2, 0, 1).float()
                frame_group.append(frame.unsqueeze(0))

            except Exception as ex:
                print(f'get frame error:{ex}')
        frame_group = torch.stack(frame_group, dim=1)

        with torch.no_grad():

            if cuda_flag:
                frame_group = frame_group.cuda()
            out_frame_group = model(frame_group, test_mode=True)['output'].cpu()
        for k in range(0, out_frame_group.size(1)):
            img = tensor2img(out_frame_group[:, k, :, :, :])
            restore_img_dqueue.append(RestoreImageWrapper(img))
        frame_group = []

    restore_img_dqueue.append(RestoreImageWrapper(is_ended=True))

    video_save_feature.result()
    if video_reader is not None:
        video_reader.close()
    thread_pool.shutdown()
    end_time = time.time()
    print('\nAll results are saved in {}, all cost time:{:.2f}秒'.format(args.output_dir, end_time - start_time))
