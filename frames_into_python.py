import os, sys
import gi
gi.require_version('Gst', '1.0')
from gi.repository import Gst
import hydra
from omegaconf import DictConfig, OmegaConf


class Pipeline:
    def __init__(self, input_src: str = "media/in.mp4", num_buffers: int = 200, sink_name: str = "s"):
        pipeline = Gst.parse_launch(f'''
            filesrc location={input_src} num-buffers={num_buffers} !
            decodebin !
            fakesink name={sink_name}
        ''')
        pipeline.get_by_name(sink_name).get_static_pad('sink').add_probe(
            Gst.PadProbeType.BUFFER,
            self.on_frame_probe
        )

        self.pipeline = pipeline

    def on_frame_probe(self, pad, info):
        buf = info.get_buffer()
        print(f'[{buf.pts / Gst.SECOND:6.2f}]')
        return Gst.PadProbeReturn.OK

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
                Gst.debug_bin_to_dot_data(self.pipeline, Gst.DebugGraphDetails.ALL)
            )
            self.pipeline.set_state(Gst.State.NULL)



@hydra.main(version_base=None, config_path="conf", config_name="config.frames_into_python")
def my_app(cfg : DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    Gst.init()
    pipeline : Pipeline = hydra.utils.instantiate(cfg.pipeline)
    pipeline.set_playing_state()
    pipeline.loop()

if __name__ == '__main__':
    my_app()
