import torch
from omegaconf import DictConfig, OmegaConf
import hydra
import numpy as np
from gi.repository import Gst
import os
import sys
import gi
gi.require_version('Gst', '1.0')


class Pipeline:
    def __init__(
        self,
        detector,
        preprocess,
        input_src: str = "media/in.mp4",
        num_buffers: int = 200,
        frame_format: str = 'RGBA',
        pixel_bytes: int = 4,
        sink_name: str = 's',
        device: str = 'cuda'
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
        self.preprocess = preprocess
        self.device = device

        self.detector.eval().to(device)
        self.detector(torch.zeros(2, 3, 300, 300, device=device))

        pipeline.get_by_name(sink_name).get_static_pad('sink').add_probe(
            Gst.PadProbeType.BUFFER,
            self.on_frame_probe
        )
        self.pipeline = pipeline

    def on_frame_probe(self, pad, info):
        buf = info.get_buffer()
        print(f'[{buf.pts / Gst.SECOND:6.2f}]')

        image_tensor = self.buffer_to_image_tensor(buf, pad.get_current_caps())
        image_batch = image_tensor.unsqueeze(0).to(self.device)
        with torch.no_grad():
            detections = self.detector(image_batch)[0]

        return Gst.PadProbeReturn.OK

    def buffer_to_image_tensor(self, buf, caps):
        caps_structure = caps.get_structure(0)
        height, width = caps_structure.get_value(
            'height'), caps_structure.get_value('width')

        is_mapped, map_info = buf.map(Gst.MapFlags.READ)
        if is_mapped:
            try:
                image_array = np.ndarray(
                    (height, width, self.pixel_bytes),
                    dtype=np.uint8,
                    buffer=map_info.data
                ).copy()  # extend array lifetime beyond subsequent unmap
                return self.preprocess(image_array[:, :, :3])  # RGBA -> RGB
            finally:
                buf.unmap(map_info)

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
            open(f'logs/{os.path.splitext(sys.argv[0])[0]}.pipeline.dot', 'w').write(
                Gst.debug_bin_to_dot_data(
                    self.pipeline, Gst.DebugGraphDetails.ALL)
            )
            self.pipeline.set_state(Gst.State.NULL)


@hydra.main(version_base=None, config_path="conf", config_name="config.frames_into_pytorch")
def my_app(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    Gst.init()
    pipeline: Pipeline = hydra.utils.instantiate(cfg.pipeline)
    pipeline.set_playing_state()
    pipeline.loop()


if __name__ == "__main__":
    my_app()
