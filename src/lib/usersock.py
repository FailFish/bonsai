import pickle
import struct
import socket

# send msg to receiver_sock
def send_msg(receiver_sock, msg):
    # prefix each message with a 4-byte length in network byte order
    msg = pickle.dumps(msg)
    msglen = len(msg)
    msg = struct.pack(">I", msglen) + msg
    receiver_sock.sendall(msg)
    return msglen


# recv msg from sender_sock
def recv_msg(sender_sock):
    # read message length and unpack it into an integer
    raw_msglen = recvall(sender_sock, 4)
    if not raw_msglen:
        return None
    msglen = struct.unpack(">I", raw_msglen)[0]
    # read the message data
    msg = recvall(sender_sock, msglen)
    msg = pickle.loads(msg)
    return msg, msglen


# recv till len(received data) == msglen from sender_sock
def recvall(sender_sock, msglen):
    # helper function to receive n bytes or return None if EOF is hit
    data = b""
    while len(data) < msglen:
        packet = sender_sock.recv(msglen - len(data))
        if not packet:
            return None
        data += packet
    return data


class User:
    def __init__(self, host, port):
        self.sock = socket.socket()
        self.host = host
        self.port = port


class Client(User):
    def __init__(self, host, port):
        super(Client, self).__init__(host, port)

    # connect to server
    def connect(self):
        self.sock.connect((self.host, self.port))

    # send to server
    def send(self, msg):
        return send_msg(self.sock, msg)

    # receive from server
    def recv(self):
        return recv_msg(self.sock)

    def training_prep(self, total_batch):
        epochs, _ = self.recv()
        # TODO: set up client dataloader
        msg = total_batch
        self.send(msg)

        return epochs


class Server(User):
    def __init__(self, host, port, nclient):
        super(Server, self).__init__(host, port)
        self.nclient = nclient
        self.clients = []
        self.sock.bind((host, port))
        self.sock.listen(nclient)
        self.sendlog = [[] for _ in range(nclient)]
        self.recvlog = [[] for _ in range(nclient)]
        self.batchsizes = []
        self.epochs = 0

    def accept_clients(self):
        for _ in range(self.nclient):
            csock, addr = self.sock.accept()
            print("Conected with", addr)
            self.clients.append(csock)
        return self.clients

    def training_prep(self, epochs):
        self.epochs = epochs
        # initialization
        for i, csock in enumerate(self.clients):
            # step 1: broadcast epochs
            dlen = send_msg(csock, self.epochs)
            self.sendlog[i].append(dlen)

            # step 2: collect batchsize, which is len(dataloader).
            batchsize, dlen = recv_msg(csock)
            self.batchsizes.append(batchsize)

        return self.batchsizes

    def get_transmission_log(self):
        return self.sendlog, self.recvlog
