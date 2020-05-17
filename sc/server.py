#!-*- encoding: utf8 -*-
"""
author: Lin Zhou
date: 2018-01-22
Local http-server for system response
"""

from http.server import HTTPServer, BaseHTTPRequestHandler, SimpleHTTPRequestHandler
from socketserver import ThreadingMixIn
from urllib.parse import urlparse, parse_qs
from urllib.parse import unquote
import os
import threading
import json
import traceback
import logging
from sc.signal_manager import SignalMgr
from sc.deep_predict import DeepPredictor
import numpy as np

logger = logging.getLogger('server')
logger.setLevel(logging.DEBUG)
file_handler = logging.FileHandler('server.log')
logger.addHandler(file_handler)

"""
https://pymotw.com/2/BaseHTTPServer/
"""
# class Handler(BaseHTTPRequestHandler):
class Handler(BaseHTTPRequestHandler):
    deep_predictor = DeepPredictor(os.path.join('models', 'deep_model', '-8729.meta'), os.path.join('models', 'deep_model'))
    def get_params_(self):
        query_components = parse_qs(urlparse(unquote(self.path)).query)
        mode = urlparse(unquote(self.path)).path.strip('/')
        query_components['mode'] = mode
        return query_components

    def do_GET(self):
        self.send_response(200)
        self.send_header('Content-Type', 'application/json')
        self.end_headers()
        jsonRetParam = dict()
        jsonRetParam['errorCode'] = 0
        jsonRetParam['resultCode'] = 0
        jsonRetParam['speed'] = 0
        jsonRetParam['speedResult'] = 0
        try:
            params = self.get_params_()
            signal_mgr = SignalMgr()
            pred_ret = signal_mgr.process(params['filepath'][0], params)

            if ('mode' not in params) or params['mode'] != 'speed':
                # deep_predictor
                dt, raw_signals = signal_mgr.parse_signals_from_file(params['filepath'][0], 1)
                deep_features = signal_mgr.get_features(params['filepath'][0], request_param={'skip_row': [1], 'model_path': ['train']})
                deep_score = self.deep_predictor.predict(raw_signals, deep_features, 4)
                deep_wave_result = np.argmax(deep_score)
            else:
                deep_wave_result = 0

            #print pred_ret
            # merge version
            # jsonRetParam['resultCode'] = max(pred_ret['stat'], deep_wave_result)
            jsonRetParam['resultCode'] = max(int(deep_wave_result), pred_ret['stat'])
            jsonRetParam['speed'] = pred_ret['speed']
            jsonRetParam['reason'] = pred_ret['reason']
            jsonRetParam['speedResult'] = pred_ret['speedResult']
            jsonRetParam['waveResult'] = max(pred_ret['waveResult'], int(deep_wave_result))
            # jsonRetParam['waveResult'] = int(deep_wave_result)
            jsonRetParam['waveScore'] = pred_ret['waveScore']
        except:
            traceback.print_exc()
            jsonRetParam['errorCode'] = 1
            jsonRetParam['speed'] = 0
            jsonRetParam['reason'] = -1
            jsonRetParam['speedResult'] = 1
            jsonRetParam['waveResult'] = 1
            jsonRetParam['waveScore'] = 0
        # print(jsonRetParam)
        self.wfile.write(json.dumps(jsonRetParam).encode())

def run():
    port = 8000
    print('starting server, port', port)
    server_address = ('', 8000)
    httpd = HTTPServer(server_address, Handler)
    print ('running server...')
    httpd.serve_forever()

# using ForkingMixIn instead of ThreadingMixIn
class ThreadedHTTPServer(ThreadingMixIn, HTTPServer):
    """Handle requests in a separate thread."""
    allow_reuse_address = True

    def shutdown(self):
        self.socket.close()
        HTTPServer.shutdown(self)


class Listener(threading.Thread):

    def __init__(self, i):
        threading.Thread.__init__(self)
        self.i = i
        self.daemon = True
        print("start: ", i)
        self.start()

    def run(self):
        server_address = ('', 8000 + self.i) # How to attach all of them to 8000?
        httpd = HTTPServer(server_address, Handler)
        httpd.serve_forever()


if __name__ == '__main__':
    #server = ThreadedHTTPServer(('localhost', 8000), Handler)
    #print 'Starting server, use <Ctrl-C> to stop'
    #server.serve_forever()

    # [ Listener(i) for i in range(8) ]
    # while True:
    #     time.sleep(1000)
    run()
