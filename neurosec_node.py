from os import path
from flask import request, jsonify
from functools import wraps
from flask import Flask, render_template, Response

from neurosec import Neurosec


class NeurosecNode:

    CERT = "neurosec_node_cert.pem"
    PRIVKEY = "neurosec_node_privkey.pem"

    def __init__(self, *args, **kwargs):

        self.node_key = kwargs.get("node_key", False)
        self.host = kwargs.get("host", False)

        self.neurosec = Neurosec(
            source=kwargs.get("source", 0),
            stream_mode=kwargs.get("stream_mode", False),
            model=kwargs.get("model"),
        ).start()
        self.app = Flask(__name__)

    def generate_stream(self, rendered):
        while True:
            jpeg = self.neurosec.get_jpeg(rendered=rendered)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n\r\n"
            )

    def check_auth(self):
        headers = request.headers

        x_node_key = headers.get("X-Node-Key", False)
        node_key = request.args.get("node_key", default="", type=str)

        if self.node_key and node_key and node_key == self.node_key:
            return True
        elif self.node_key and x_node_key and x_node_key == self.node_key:
            return True
        else:
            return False

    def run(self):
        def auth_required(func):
            @wraps(func)
            def wrap(*args, **kwargs):
                if self.check_auth():
                    return func(*args, **kwargs)
                else:
                    return jsonify({"message": "ERROR: Unauthorized"}), 401

            return wrap

        @self.app.route("/")
        @auth_required
        def index():
            context = {
                "node_key": self.node_key,
            }
            return render_template(
                path.join(path.dirname(__file__), "templates/index.html"),
                **context
            )

        @self.app.route("/feed/")
        @auth_required
        def video_feed():
            return Response(
                self.generate_stream(False),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

        @self.app.route("/feed/overlay/")
        @auth_required
        def video_feed_with_overlay():
            return Response(
                self.generate_stream(True),
                mimetype="multipart/x-mixed-replace; boundary=frame",
            )

        @self.app.route("/meta")
        @auth_required
        def video_meta():
            frame = self.neurosec.read()
            inference = self.neurosec.get_frame_meta_json(frame)
            return jsonify(inference)

        self.app.run(
            host=self.host,
            debug=False,
        )
