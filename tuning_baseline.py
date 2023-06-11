import torch
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
        model_precision: str = "fp32"
    ):
        pipeline = Gst.parse_launch(f'''
            filesrc location={input_src} num-buffers={num_buffers} !
            decodebin !
            nvvideoconvert !
            video/x-raw,format={frame_format} !
            fakesink name={sink_name}
        ''')
        self.pixel_bytes = pixel_bytes
        self.detector = detector
        self.ssd_utils = ssd_utils
        self.device = device
        self.detection_threshold = detection_threshold
        self.model_precision = model_precision
        self.model_dtype = torch.float16 if model_precision == 'fp16' else torch.float32
        self.start_time = None
        self.frames_processed = 0

        self.detector.eval().to(device)
        self.detector(torch.zeros(1,3,300,300,device=device))

        pipeline.get_by_name(sink_name).get_static_pad('sink').add_probe(
            Gst.PadProbeType.BUFFER,
            self.on_frame_probe
        )
        self.pipeline = pipeline

    def on_frame_probe(self, pad, info):
        self.start_time = self.start_time or time.time()

        with nvtx_range('on_frame_probe'):
            buf = info.get_buffer()

            image_tensor = self.buffer_to_image_tensor(buf, pad.get_current_caps())
            image_batch = self.preprocess(image_tensor.unsqueeze(0))
            self.frames_processed += image_batch.size(0)

            print(f'[{buf.pts / Gst.SECOND:6.2f}]', "frames", self.frames_processed)

            with torch.no_grad():
                with nvtx_range('inference'):
                    locs, labels = self.detector(image_batch)
                self.postprocess(locs, labels)

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


    def postprocess(self, locs, labels):
        with nvtx_range('postprocess'):
            results_batch = self.ssd_utils.decode_results((locs, labels))
            results_batch = [self.ssd_utils.pick_best(results, self.detection_threshold) for results in results_batch]
            for bboxes, classes, scores in results_batch:
                if scores.shape[0] > 0:
                    print(bboxes, classes, scores)

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


@hydra.main(version_base=None, config_path="conf", config_name="tuning_postprocess_1")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    Gst.init()
    pipeline: Pipeline = hydra.utils.instantiate(cfg.pipeline)
    print("precision", pipeline.model_precision)
    pipeline.set_playing_state()
    pipeline.loop()


if __name__ == "__main__":
    my_app()
