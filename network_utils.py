import socket
import pickle
import struct
import torch
import time
from config import CMD

def send_message(conn, cmd, data=None):
    if data is not None:
        data_bytes = pickle.dumps(data)
    else:
        data_bytes = b''

    conn.sendall(struct.pack('B', cmd))
    
    conn.sendall(struct.pack('>I', len(data_bytes)))
    
    if data_bytes:
        conn.sendall(data_bytes)

def recv_message(conn):
    cmd_bytes = conn.recv(1)
    if not cmd_bytes:
        raise ConnectionError("Connection closed")
    cmd = struct.unpack('B', cmd_bytes)[0]
    
    length_bytes = conn.recv(4)
    if not length_bytes:
        raise ConnectionError("Connection closed")
    data_length = struct.unpack('>I', length_bytes)[0]
    
    if data_length > 0:
        data_bytes = b''
        while len(data_bytes) < data_length:
            chunk = conn.recv(min(data_length - len(data_bytes), 65536))
            if not chunk:
                raise ConnectionError("Connection closed during data transfer")
            data_bytes += chunk
        data = pickle.loads(data_bytes)
    else:
        data = None
    
    return cmd, data

def send_tensor(conn, tensor):
    data = tensor.cpu().detach()
    send_message(conn, CMD.FORWARD_DATA, data)

def recv_tensor(conn):
    cmd, data = recv_message(conn)
    return data

class ServerConnection:
    def __init__(self, port):
        self.port = port
        self.socket = None
        self.conn = None
        self.addr = None
    
    def start(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        self.socket.bind(('0.0.0.0', self.port))
        self.socket.listen(1)
        print(f"[Server] Listening on port {self.port}...")
        self.conn, self.addr = self.socket.accept()
        print(f"[Server] Client connected: {self.addr}")
        return self.conn
    
    def send(self, cmd, data=None):
        send_message(self.conn, cmd, data)
    def recv(self):
        return recv_message(self.conn)
    def close(self):
        if self.conn:
            try:
                self.send(CMD.SHUTDOWN)
            except:
                pass
            self.conn.close()
        if self.socket:
            self.socket.close()
        print("[Server] Connection closed")

class ClientConnection:
    def __init__(self, server_ip, port):
        self.server_ip = server_ip
        self.port = port
        self.socket = None
    
    def connect(self, max_retries=30, retry_interval=2):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        for i in range(max_retries):
            try:
                print(f"[Client] Connecting to {self.server_ip}:{self.port}... (attempt {i+1}/{max_retries})")
                self.socket.connect((self.server_ip, self.port))
                print(f"[Client] Connected to server")
                return self.socket
            except ConnectionRefusedError:
                if i < max_retries - 1:
                    time.sleep(retry_interval)
                else:
                    raise ConnectionError(f"Failed to connect after {max_retries} attempts")
    
    def send(self, cmd, data=None):
        send_message(self.socket, cmd, data)
    def recv(self):
        return recv_message(self.socket)
    def close(self):
        if self.socket:
            self.socket.close()
        print("[Client] Connection closed")