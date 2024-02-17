# Copyright (c) Microsoft Corporation.
# SPDX-License-Identifier: Apache-2.0

# DeepSpeed Team
import asyncio
import time
import threading
import sys
import os
import signal

from flask import Flask, request, jsonify
from flask_restful import Resource, Api
from werkzeug.serving import make_server

import mii
from mii.constants import RESTFUL_GATEWAY_SHUTDOWN_TIMEOUT, RESTFUL_API_PATH

GATEWAY_THREAD = None

def init_gateway(deployment_name, rest_host, rest_port, rest_procs):
    global GATEWAY_THREAD
    if GATEWAY_THREAD is None:
        GATEWAY_THREAD = RestfulGatewayThread(deployment_name, rest_host, rest_port, rest_procs)
    return GATEWAY_THREAD

# Signal handler function
def signal_handler(signum, frame):
    global GATEWAY_THREAD
    GATEWAY_THREAD.server.shutdown()

# Set up signal handling
signal.signal(signal.SIGUSR1, signal_handler)

def createRestfulGatewayApp(deployment_name, server_thread):
    class RestfulGatewayService(Resource):
        def __init__(self):
            super().__init__()
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            self.client = mii.client(deployment_name)

        def post(self):
            data = request.get_json()
            result = self.client.generate(**data)
            return jsonify([r.to_msg_dict() for r in result])

    app = Flask("RestfulGateway")

    @app.route("/terminate", methods=["GET"])
    def terminate():
        # Send parent process a signal
        os.kill(os.getppid(), signal.SIGUSR1)
        return "Shutting down RESTful API gateway server"

    api = Api(app)
    path = "/{}/{}".format(RESTFUL_API_PATH, deployment_name)
    api.add_resource(RestfulGatewayService, path)

    return app


class RestfulGatewayThread(threading.Thread):
    def __init__(self, deployment_name, rest_host, rest_port, rest_procs):
        threading.Thread.__init__(self)
        self.deployment_name = deployment_name

        app = createRestfulGatewayApp(deployment_name, self)
        self.server = make_server(rest_host,
                                  rest_port,
                                  app,
                                  threaded=False,
                                  processes=rest_procs)
        self.ctx = app.app_context()
        self.ctx.push()

    def run(self):
        self.server.serve_forever()
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        client = mii.client(self.deployment_name)
        try:
            loop.run_until_complete(client.terminate_async())
        except:
            pass
