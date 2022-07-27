# Neurosec

A neuromorphic inference wrapper for the popular VidGear video processing library. Neurosec seamlessly allows you to process inference using the Akida neuromorphic processor.

For best results, ensure you have an Akida neuromorphic processor installed. If you do not have one, you can purchase one from [Brainchip Inc.](https://shop.brainchipinc.com/):



### How to install

Install via pip:

```bash
pip install neurosec

```


### How to use Neurosec

Here is a simple example of using Neurosec to display a stream from a camera and render an overlay of detected objects

```python

import cv2
from neurosec import Neurosec


yolo_face = {
    "fbz": "models/yolo_face.fbz",
    "predict_classes": False,
    "anchors": [[0.90751, 1.49967], [1.63565, 2.43559], [2.93423, 3.88108]],
    "classes": 1,
    "labels": {
        0: "face",
    },
    "colours": {0: (255, 0, 0)},
    "pred_conf_min": 0.70,
}

neurosec = Neurosec(
    source=0,
    model=yolo_face,
    resolution=(640, 480),
).start()

while True:
    frame = neurosec.get_neurosec_frame()
    if frame is None:
        break

    cv2.imshow("Output", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == ord("q"):
        break
```

Since Neurosec leans entirely on [VidGear](https://github.com/abhiTronix/vidgear) - all of the amazing options are available, like streaming from youtube.

```python

from neurosec import Neurosec

neurosec = Neurosec(
    source="https://www.youtube.com/watch?v=crddAe9N2aM",
    stream_mode=True,
    model={
        "fbz": "models/yolo.fbz",
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
    },
).start()
```

A complete example:

```python

import cv2
from neurosec import Neurosec

if __name__ == "__main__":
    try:

        neurosec = Neurosec(
            source=0,
            model={
                "fbz": "models/yolo.fbz",
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
            },
            resolution=(640, 480),
        ).start()

        while True:
            frame = neurosec.get_neurosec_frame()
            if frame is None:
                break

            cv2.imshow("Output", frame)

            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    except KeyboardInterrupt:
        print("die")
```

You can get frame meta at any time by calling: 

```python

from neurosec import Neurosec

neurosec = Neurosec(
            source=0,
            model={
                "fbz": "models/yolo_face.fbz",
                "predict_classes": False,
                "anchors": [
                    [0.90751, 1.49967],
                    [1.63565, 2.43559],
                    [2.93423, 3.88108],
                ],
                "classes": 1,
                "labels": {
                    0: "face",
                },
                "colours": {0: (255, 255, 0), 1: (255, 0, 0)},
                "pred_conf_min": 0.70,
            },
            resolution=(640, 480),
        ).start()


meta = neurosec.get_frame_meta_json()
```


# NeurosecNode

A simple Flask based web app that provides four main endpoints:

1. {host} # generic view with streaming video embed
1. {host}/feed/ # streaming video
1. {host}/feed/overlay/  # streaming video with overlay
1. {host}/meta/ # frame meta


### Accessing the server

An example request trying to access the nodes IP:

Through your browser:

Go to your computer or the devices IP address: `http://10.0.0.1:5000?node_key={your_key}`


### Example meta output

```python

import requests

your_node_key = "abcdefg"
url = "http://10.0.0.1:5000/meta"
headers = {"X-Node-Key": your_node_key}

meta = requests.get(url, headers=headers)

print(meta.json())
```

or visit: `http://10.0.0.1:5000/meta?node_key={your_key}`

An example output while running Yolo:

```json
{"timestamp": "1657216671.52087", "original_resolution": [640, 480], "decoded": [[231.37685351750486, 95.64780969570069, 434.0158447009765, 340.4589660097876, 0.0, 0.9620453119277954]], "labels": {"0": "face"}, "colours": {"0": [255, 0, 0]}, "pred_conf_min": 0.7}
```


### How to run Neurosec-node

An example to start a server that will stream from the camera located at /dev/video0

```python

from neurosec import NeurosecNode


neurosec_node = NeurosecNode(
    **{
        "source": 0,
        "resolution": (640, 480),
        "host": "0.0.0.0",
        "node_key": "this_is_a_passw0rd",
        "model": {
            "fbz": "models/yolo_face.fbz",
            "predict_classes": False,
            "anchors": [
                [0.90751, 1.49967],
                [1.63565, 2.43559],
                [2.93423, 3.88108],
            ],
            "classes": 1,
            "labels": {
                0: "face",
            },
            "colours": {0: (255, 0, 0)},
            "pred_conf_min": 0.70,
        },
    }
).run()
```
