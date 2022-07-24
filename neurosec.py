import io
import time
import uuid
import json
from json import JSONEncoder
import cv2
from PIL import Image
import numpy as np
import akida
from akida import Model, devices
from akida_models.detection import processing
from vidgear.gears import VideoGear


class NumpyArrayEncoder(JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        return JSONEncoder.default(self, obj)


class Inference:

    """

    Class to receive input data and run inference over it. If there is
    and Akida device available, it will choose the first akida.NSoC_v2

    usage:

    model = {
        "fbz": YOLO,
        "predict_classes": False,
        "anchors": [
            [0.56615, 1.05064],
            [1.09098, 2.04053],
            [2.384, 3.00597],
            [2.45964, 4.91562],
            [5.16724, 5.56961],
        ],
        "classes": 2,
        "labels": {
            0: "car",
            1: "person",
        },
        "colours": {0: (255, 255, 0), 1: (255, 0, 0)},
        "pred_conf_min": 0.70,
    }

    inference = Inference(power_measurement=False, model=model)

    """

    def __init__(self, *args, **kwargs):

        self.input_data = []
        self.output_data = []

        self.power_measurement = kwargs.get("power_measurement", False)
        self.device = self.get_device()
        self.mapped = False

        self.akida_model = False

        model = kwargs.get("model", False)
        if model and type(model) is dict:
            self.model = model

        filename = self.model.get("fbz", False)
        predict_classes = self.model.get("predict_classes")
        if filename:
            akida_model = {
                "model": Model(filename=filename),
                "predict_classes": predict_classes,
            }
            self.akida_model = akida_model

            # map akida model to hardware
            if self.device and self.mapped is False:
                self.akida_model["model"].map(self.device, hw_only=True)
                self.mapped = True

    def get_device(self):
        for device in devices():
            if device.version == akida.NSoC_v2:
                device.soc.power_measurement_enabled = self.power_measurement
                return device

        return False

    def preprocess_image(self, input_data, input_shape):
        input_arr = processing.preprocess_image(
            input_data, (input_shape[0], input_shape[1])
        )
        input_arr = np.array([input_arr], dtype="uint8")
        return input_arr

    def decode_output(self, pots, original_resolution, anchors, classes):

        w, h, c = pots.shape
        pots = pots.reshape((h, w, len(anchors), 4 + 1 + classes))

        raw_boxes = processing.decode_output(pots, anchors, 1)

        pred_boxes = np.array(
            [
                [
                    box.x1 * original_resolution[0],
                    box.y1 * original_resolution[1],
                    box.x2 * original_resolution[0],
                    box.y2 * original_resolution[1],
                    box.get_label(),
                    box.get_score(),
                ]
                for box in raw_boxes
            ]
        )

        return pred_boxes

    def process_data(self, data):

        timestamp = data.get("timestamp", False)
        input_data = data.get("input", False)
        original_resolution = data.get("original_resolution", False)

        if self.akida_model and timestamp and type(input_data) is np.ndarray:

            predictions = {}

            akida_model = self.akida_model["model"]

            if type(self.akida_model["model"] is akida.core.Model):

                akida_model = self.akida_model["model"]

                input_array = self.preprocess_image(
                    input_data, akida_model.input_shape
                )

                if self.model.get("predict_classes", False) is False:

                    p = akida_model.predict(input_array)[0]

                    data["decoded"] = self.decode_output(
                        p,
                        original_resolution,
                        self.model["anchors"],
                        self.model["classes"],
                    )
                    data["labels"] = self.model["labels"]
                    data["colours"] = self.model["colours"]
                    data["pred_conf_min"] = self.model["pred_conf_min"]
                    predictions = p
                else:
                    p = akida_model.predict(input_array)
                    predictions = p

                data["predictions"] = predictions

        return data


class Neurosec(VideoGear):

    DEFAULT_RESOLUTION = (640, 480)

    def __init__(self, *args, **kwargs):

        self.model = kwargs.pop("model")
        self.device = False
        self.uuid = uuid.uuid4().__str__()
        self.inference = Inference(model=self.model)

        self.original_resolution = kwargs.get(
            "resolution", self.DEFAULT_RESOLUTION
        )

        super(Neurosec, self).__init__(*args, **kwargs)

        if kwargs.get("stream_mode", False) is True:
            self.original_resolution = self.get_resolution()

    def get_resolution(self):
        resolution = self.stream.ytv_metadata["resolution"].split("x")
        return (int(resolution[0]), int(resolution[1]))

    def get_jpeg(self, rendered=False):

        if rendered:
            frame = self.get_neurosec_frame()
        else:
            frame = self.read()

        img = Image.fromarray(frame)
        jpeg = io.BytesIO()
        img.save(jpeg, format="jpeg")

        return jpeg.getvalue()

    def get_frame_meta_json(
        self, frame, include_frame=False, include_predictions=False
    ):
        frame = self.read()
        frame_meta = self.get_frame_meta(frame)
        if include_frame is False:
            frame_meta.pop("input")
        if include_predictions is False:
            frame_meta.pop("predictions")
        frame_meta_json = json.dumps(frame_meta, cls=NumpyArrayEncoder)
        return frame_meta_json

    def get_frame_meta(self, frame):

        frame_meta = {}

        if type(frame) is np.ndarray:

            data = {
                "timestamp": str(time.time()),
                "input": frame,
                "original_resolution": self.original_resolution,
            }
            frame_meta = self.inference.process_data(data)

        return frame_meta

    def get_neurosec_frame(self):
        frame = self.read()
        frame_meta = self.get_frame_meta(frame)

        if type(frame_meta) is dict:
            boxes = frame_meta.get("decoded")
            if type(boxes) is np.ndarray:
                frame = self.render_boxes(frame, boxes, frame_meta)
        return frame

    def render_boxes(self, frame, boxes, frame_meta):

        labels = frame_meta["labels"]
        colours = frame_meta["colours"]

        if type(boxes) is np.ndarray:

            for box in boxes:
                if box[5] > frame_meta["pred_conf_min"]:
                    x1, y1 = int(box[0]), int(box[1])
                    x2, y2 = int(box[2]), int(box[3])

                    label = "object"
                    if type(labels) is dict:
                        label = labels[int(box[4])]

                    score = "{:.2%}".format(box[5])

                    colour = (255, 255, 255)
                    if type(colours) is dict and box[4] in colours:
                        colour = colours[box[4]]

                    frame = cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 1)
                    cv2.putText(
                        frame,
                        "{} - {}".format(label, score),
                        (x1, y1 - 5),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        colour,
                        1,
                        cv2.LINE_AA,
                    )

        return frame
