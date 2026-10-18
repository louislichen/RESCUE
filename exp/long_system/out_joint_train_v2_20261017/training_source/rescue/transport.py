"""Bounded JSON TCP transport for length decisions, independent of context."""
import json
import math
import socket
import socketserver
import struct

import torch

from .messages import LengthPacket, ProtocolError, RoundKey
from .policy import choose

MAX_MESSAGE = 65536


def receive(sock):
    def exact(size):
        chunks = []
        while size:
            data = sock.recv(size)
            if not data:
                raise EOFError('Length connection closed')
            chunks.append(data)
            size -= len(data)
        return b''.join(chunks)
    size = struct.unpack('!I', exact(4))[0]
    if not 0 < size <= MAX_MESSAGE:
        raise ProtocolError('Length message exceeds frame limit')
    try:
        return json.loads(exact(size))
    except (UnicodeError, json.JSONDecodeError) as error:
        raise ProtocolError('Invalid JSON length message') from error


def send(sock, value):
    data = json.dumps(value, separators=(',', ':'), allow_nan=False).encode()
    if len(data) > MAX_MESSAGE:
        raise ProtocolError('Length message exceeds frame limit')
    sock.sendall(struct.pack('!I', len(data))+data)
    return len(data)+4


def validate_request(value):
    fields = {'schema', 'key', 'features', 'cost', 'remaining', 'max_draft'}
    if not isinstance(value, dict) or set(value) != fields or type(value['schema']) is not int or value['schema'] != 1:
        raise ProtocolError('Invalid length request schema')
    key = RoundKey.from_dict(value['key'])
    features = value['features']
    if not isinstance(features, list) or len(features) != 31 or any(type(v) not in (int, float) or not math.isfinite(v) for v in features):
        raise ProtocolError('Expected 31 finite features')
    if type(value['remaining']) is not int or value['remaining'] < 1 or type(value['max_draft']) is not int or not 1 <= value['max_draft'] <= 12:
        raise ProtocolError('Invalid horizon/output budget')
    cost = value['cost']
    if not isinstance(cost, dict) or set(cost)-{'verify_beta', 'student_step', 'cache_update', 'fixed_round'}:
        raise ProtocolError('Invalid cost fields')
    beta = cost.get('verify_beta')
    if not isinstance(beta, list) or len(beta) != 4:
        raise ProtocolError('Invalid verification coefficients')
    numbers = beta+[cost.get('student_step'), cost.get('cache_update', 0.), cost.get('fixed_round', 0.)]
    if any(type(v) not in (int, float) or not math.isfinite(v) or v < 0 for v in numbers) or cost['student_step'] <= 0:
        raise ProtocolError('Invalid timing model')
    return key


class LengthClient:
    def __init__(self, endpoint, timeout=1.):
        self.endpoint, self.timeout, self.socket = endpoint, timeout, None
        self.bytes_sent = 0
        self.calls = 0

    def choose(self, key, vector, cost, remaining, max_draft):
        try:
            if self.socket is None:
                self.socket = socket.create_connection(self.endpoint, self.timeout)
                self.socket.settimeout(self.timeout)
                self.socket.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
            request = dict(schema=1, key=key.to_dict(), features=vector.detach().cpu().tolist(),
                           cost={k: cost.get(k, 0.) for k in ('student_step', 'cache_update', 'fixed_round')},
                           remaining=remaining, max_draft=max_draft)
            request['cost']['verify_beta'] = cost['verify_beta']
            self.bytes_sent += send(self.socket, request)
            response = receive(self.socket)
            self.calls += 1
            if not isinstance(response, dict) or set(response) != {'schema', 'key', 'cap', 'score'} or response['schema'] != 1:
                raise ProtocolError('Invalid length response schema')
            return LengthPacket(RoundKey.from_dict(response['key']), response['cap'], response['score']).validate(key, min(remaining, max_draft))
        except (OSError, EOFError, ProtocolError):
            self.close()  # A timed-out response can never be read as a later round.
            raise

    def close(self):
        if self.socket is not None:
            self.socket.close()
            self.socket = None


class LengthServer(socketserver.ThreadingTCPServer):
    allow_reuse_address = True
    daemon_threads = True

    def __init__(self, address, model):
        self.model = model
        super().__init__(address, LengthHandler)


class LengthHandler(socketserver.BaseRequestHandler):
    def handle(self):
        self.request.settimeout(30.)
        self.request.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        while True:
            try:
                request = receive(self.request)
                key = validate_request(request)
                cap, score = choose(self.server.model, torch.tensor(request['features']), request['cost'],
                                    key.prefix, request['remaining'], request['max_draft'])
                send(self.request, dict(schema=1, key=key.to_dict(), cap=cap, score=score))
            except (EOFError, OSError, ProtocolError):
                return
