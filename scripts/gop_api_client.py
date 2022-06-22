import argparse
import json
import ws4py.messaging
from ws4py.client.threadedclient import WebSocketClient
import time


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--input', type=str, help='Input wav file')
    parser.add_argument('--text', type=str, help='Text')
    parser.add_argument('--url', type=str)
    return parser.parse_args()


class Client(WebSocketClient):
    def __init__(self, url: str, input_path: str, text: str):
        super().__init__(url)
        with open(input_path, 'rb') as f:
            self.wav = f.read()
        self.text = text

    def received_message(self, msg):
        assert isinstance(msg, ws4py.messaging.TextMessage)
        msg = json.loads(msg.data.decode('utf-8'))
        print(f'Result: {msg}')

    def opened(self):
        # 1. metadata
        metadata = dict(ext='wav', transcript=self.text)
        self.send(json.dumps(metadata))

        # 2. wav signal as binary
        self.send(self.wav, True)


def main():
    args = get_args()

    ws = Client(args.url, args.input, args.text)
    ws.connect()

    while not ws.client_terminated:
        time.sleep(0.5)

    ws.close()


if __name__ == '__main__':
    main()
