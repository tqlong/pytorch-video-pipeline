import math
import itertools
import torch
import torchvision
from omegaconf import DictConfig, OmegaConf
import hydra
import numpy as np
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import os
import sys
import contextlib
import time


# context manager to help keep track of ranges of time, using NVTX
@contextlib.contextmanager
def nvtx_range(msg):
    depth = torch.cuda.nvtx.range_push(msg)
    try:
        yield depth
    finally:
        torch.cuda.nvtx.range_pop()


class Pipeline:
    def __init__(
        self,
        detector,
        ssd_utils,
        input_src: str = "media/in.mp4",
        num_buffers: int = 200,
        frame_format: str = 'RGBA',
        pixel_bytes: int = 4,
        sink_name: str = 's',
        device: str = 'cuda',
        detection_threshold: float = 0.4,
        model_precision: str = "fp32",
        batch_size: int = 4
    ):
        pipeline = self.create_pipeline(input_src, num_buffers, frame_format, sink_name)
        self.pixel_bytes = pixel_bytes
        self.detector = detector
        self.ssd_utils = ssd_utils
        self.device = device
        self.detection_threshold = detection_threshold
        self.model_precision = model_precision
        self.model_dtype = torch.float16 if model_precision == 'fp16' else torch.float32
        self.start_time = None
        self.frames_processed = 0
        self.dboxes_xywh = self.init_dboxes().unsqueeze(dim=0)
        self.scale_xy = 0.1
        self.scale_wh = 0.2
        self.image_batch, self.batch_size = [], batch_size

        self.detector.eval().to(device)
        self.detector(torch.zeros(1,3,300,300,device=device,dtype=self.model_dtype))

        pipeline.get_by_name(sink_name).get_static_pad('sink').add_probe(
            Gst.PadProbeType.BUFFER,
            self.on_frame_probe
        )
        self.pipeline = pipeline
    
    def create_pipeline(self, input_src, num_buffers, frame_format, sink_name):
        return Gst.parse_launch(f'''
            filesrc location={input_src} num-buffers={num_buffers} !
            decodebin !
            nvvideoconvert !
            video/x-raw,format={frame_format} !
            fakesink name={sink_name}
        ''')

    def on_frame_probe(self, pad, info):
        self.start_time = self.start_time or time.time()

        if not self.image_batch:
            torch.cuda.nvtx.range_push('batch')
            torch.cuda.nvtx.range_push('create_batch')

        buf = info.get_buffer()
        print(f'[{buf.pts / Gst.SECOND:6.2f}]')

        image_tensor = self.buffer_to_image_tensor(buf, pad.get_current_caps())
        self.image_batch.append(image_tensor)

        if len(self.image_batch) < self.batch_size:
            return Gst.PadProbeReturn.OK

        torch.cuda.nvtx.range_pop() # create_batch

        image_batch = self.preprocess(torch.stack(self.image_batch))
        self.frames_processed += image_batch.size(0)

        with torch.no_grad():
            with nvtx_range('inference'):
                locs, labels = self.detector(image_batch)
                self.image_batch = []
            self.postprocess(locs, labels)

        torch.cuda.nvtx.range_pop() # batch
        return Gst.PadProbeReturn.OK

    def buffer_to_image_tensor(self, buf, caps):
        with nvtx_range('buffer_to_image_tensor'):
            caps_structure = caps.get_structure(0)
            height, width = caps_structure.get_value('height'), caps_structure.get_value('width')

            is_mapped, map_info = buf.map(Gst.MapFlags.READ)
            if is_mapped:
                try:
                    image_array = np.ndarray(
                        (height, width, self.pixel_bytes),
                        dtype=np.uint8,
                        buffer=map_info.data
                    )
                    return torch.from_numpy(
                        image_array[:,:,:3].copy() # RGBA -> RGB, and extend lifetime beyond subsequent unmap
                    )
                finally:
                    buf.unmap(map_info)


    def preprocess(self, image_batch):
        '300x300 centre crop, normalize, HWC -> CHW'
        with nvtx_range('preprocess'):
            batch_dim, image_height, image_width, image_depth = image_batch.size()
            copy_x, copy_y = min(300, image_width), min(300, image_height)

            dest_x_offset = max(0, (300 - image_width) // 2)
            source_x_offset = max(0, (image_width - 300) // 2)
            dest_y_offset = max(0, (300 - image_height) // 2)
            source_y_offset = max(0, (image_height - 300) // 2)

            input_batch = torch.zeros((batch_dim, 300, 300, 3), dtype=self.model_dtype, device=self.device)
            input_batch[:, dest_y_offset:dest_y_offset + copy_y, dest_x_offset:dest_x_offset + copy_x] = \
                image_batch[:, source_y_offset:source_y_offset + copy_y, source_x_offset:source_x_offset + copy_x]

            return torch.einsum(
                'bhwc -> bchw',
                self.normalize(input_batch / 255)
            ).contiguous()


    def normalize(self, input_tensor):
        'Nvidia SSD300 code uses mean and std-dev of 128/256'
        return (2.0 * input_tensor) - 1.0

    def init_dboxes(self):
        'adapted from https://github.com/NVIDIA/DeepLearningExamples/blob/master/PyTorch/Detection/SSD/src/utils.py'
        fig_size = 300
        feat_size = [38, 19, 10, 5, 3, 1]
        steps = [8, 16, 32, 64, 100, 300]
        scales = [21, 45, 99, 153, 207, 261, 315]
        aspect_ratios = [[2], [2, 3], [2, 3], [2, 3], [2], [2]]

        fk = fig_size / torch.tensor(steps).float()

        dboxes = []
        # size of feature and number of feature
        for idx, sfeat in enumerate(feat_size):
            sk1 = scales[idx] / fig_size
            sk2 = scales[idx + 1] / fig_size
            sk3 = math.sqrt(sk1 * sk2)
            all_sizes = [(sk1, sk1), (sk3, sk3)]

            for alpha in aspect_ratios[idx]:
                w, h = sk1 * math.sqrt(alpha), sk1 / math.sqrt(alpha)
                all_sizes.append((w, h))
                all_sizes.append((h, w))

            for w, h in all_sizes:
                for i, j in itertools.product(range(sfeat), repeat=2):
                    cx, cy = (j + 0.5) / fk[idx], (i + 0.5) / fk[idx]
                    dboxes.append((cx, cy, w, h))

        return torch.tensor(
            dboxes,
            dtype=self.model_dtype,
            device=self.device
        ).clamp(0, 1)

    def xywh_to_xyxy(self, bboxes_batch, scores_batch):
        bboxes_batch = bboxes_batch.permute(0, 2, 1)
        scores_batch = scores_batch.permute(0, 2, 1)

        bboxes_batch[:, :, :2] = self.scale_xy * bboxes_batch[:, :, :2]
        bboxes_batch[:, :, 2:] = self.scale_wh * bboxes_batch[:, :, 2:]

        bboxes_batch[:, :, :2] = bboxes_batch[:, :, :2] * self.dboxes_xywh[:, :, 2:] + self.dboxes_xywh[:, :, :2]
        bboxes_batch[:, :, 2:] = bboxes_batch[:, :, 2:].exp() * self.dboxes_xywh[:, :, 2:]

        # transform format to ltrb
        l, t, r, b = bboxes_batch[:, :, 0] - 0.5 * bboxes_batch[:, :, 2],\
                    bboxes_batch[:, :, 1] - 0.5 * bboxes_batch[:, :, 3],\
                    bboxes_batch[:, :, 0] + 0.5 * bboxes_batch[:, :, 2],\
                    bboxes_batch[:, :, 1] + 0.5 * bboxes_batch[:, :, 3]

        bboxes_batch[:, :, 0] = l
        bboxes_batch[:, :, 1] = t
        bboxes_batch[:, :, 2] = r
        bboxes_batch[:, :, 3] = b

        return bboxes_batch, torch.nn.functional.softmax(scores_batch, dim=-1)

    def postprocess(self, locs, labels):
        with nvtx_range('postprocess'):
            locs, probs = self.xywh_to_xyxy(locs, labels)

            # flatten batch and classes
            batch_dim, box_dim, class_dim = probs.size()
            flat_locs = locs.reshape(-1, 4).repeat_interleave(class_dim, dim=0)
            flat_probs = probs.view(-1)
            class_indexes = torch.arange(class_dim, device=self.device).repeat(batch_dim * box_dim)
            image_indexes = (torch.ones(box_dim * class_dim, device=self.device) * torch.arange(1, batch_dim + 1, device=self.device).unsqueeze(-1)).view(-1)

            # only do NMS on detections over threshold, and ignore background (0)
            threshold_mask = (flat_probs > self.detection_threshold) & (class_indexes > 0)
            flat_locs = flat_locs[threshold_mask]
            flat_probs = flat_probs[threshold_mask]
            class_indexes = class_indexes[threshold_mask]
            image_indexes = image_indexes[threshold_mask]

            nms_mask = torchvision.ops.boxes.batched_nms(
                flat_locs,
                flat_probs,
                class_indexes * image_indexes,
                iou_threshold=0.7
            )

            bboxes = flat_locs[nms_mask].cpu()
            probs = flat_probs[nms_mask].cpu()
            class_indexes = class_indexes[nms_mask].cpu()
            if bboxes.size(0) > 0:
                print(bboxes, class_indexes, probs)

    def set_playing_state(self):
        self.pipeline.set_state(Gst.State.PLAYING)

    def loop(self):
        try:
            while True:
                msg = self.pipeline.get_bus().timed_pop_filtered(
                    Gst.SECOND,
                    Gst.MessageType.EOS | Gst.MessageType.ERROR
                )
                if msg:
                    text = msg.get_structure().to_string() if msg.get_structure() else ''
                    msg_type = Gst.message_type_get_name(msg.type)
                    print(f'{msg.src.name}: [{msg_type}] {text}')
                    break
        finally:
            finish_time = time.time()
            open(f'logs/{os.path.splitext(sys.argv[0])[0]}.pipeline.dot', 'w').write(
                Gst.debug_bin_to_dot_data(self.pipeline, Gst.DebugGraphDetails.ALL)
            )
            self.pipeline.set_state(Gst.State.NULL)
            print(f'FPS: {self.frames_processed / (finish_time - self.start_time):.2f}')


@hydra.main(version_base=None, config_path="conf", config_name="tuning_batch")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    Gst.init()
    pipeline: Pipeline = hydra.utils.instantiate(cfg.pipeline)
    print("precision", pipeline.model_precision)
    pipeline.set_playing_state()
    pipeline.loop()


if __name__ == "__main__":
    my_app()
