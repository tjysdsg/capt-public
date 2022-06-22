import urllib.parse
import json
from io import BytesIO
from ws4py.client.threadedclient import WebSocketClient
import threading
import queue
import time
import sys
from gop_server import server_config


def rate_limited(max_per_sec):
    minInterval = 1.0 / float(max_per_sec)

    def decorate(func):
        lastTimeCalled = [0.0]

        def rate_limited_function(*args, **kargs):
            elapsed = time.clock() - lastTimeCalled[0]
            leftToWait = minInterval - elapsed
            if leftToWait > 0:
                time.sleep(leftToWait)
            ret = func(*args, **kargs)
            lastTimeCalled[0] = time.clock()
            return ret

        return rate_limited_function

    return decorate


class ASRClient(WebSocketClient):
    def __init__(self, audiofile, url, protocols=None, extensions=None, heartbeat_freq=None, byterate=32000,
                 save_adaptation_state_filename=None, send_adaptation_state_filename=None):
        super(ASRClient, self).__init__(url, protocols, extensions, heartbeat_freq)
        self.results = queue.Queue()
        self.trans = ''
        self.audiofile = audiofile
        self.byterate = byterate
        self.save_adaptation_state_filename = save_adaptation_state_filename
        self.send_adaptation_state_filename = send_adaptation_state_filename

    @rate_limited(4)
    def send_data(self, data):
        self.send(data, binary=True)

    def opened(self):
        def send_data_to_ws():
            if self.send_adaptation_state_filename is not None:
                try:
                    adaptation_state_props = json.load(open(self.send_adaptation_state_filename, "r"))
                    self.send(json.dumps(dict(adaptation_state=adaptation_state_props)))
                except:
                    e = sys.exc_info()[0]
                    raise NameError(f"Failed to send adaptation state: {e}")
            with self.audiofile as audiostream:
                for block in iter(lambda: audiostream.read(self.byterate // 4), b''):
                    self.send_data(block)
            self.send("EOS")

        t = threading.Thread(target=send_data_to_ws)
        t.start()

    def received_message(self, m):
        response = json.loads(str(m))
        if response['status'] == 0:
            if 'result' in response:
                trans = response['result']['hypotheses'][0]['transcript']
                self.trans = trans
            if 'adaptation_state' in response:
                if self.save_adaptation_state_filename:
                    with open(self.save_adaptation_state_filename, "w") as f:
                        f.write(json.dumps(response['adaptation_state']))
        else:
            if 'message' in response:
                msg = f"Error message: {response['message']}"
            else:
                msg = ''
            raise NameError(f"Received error from server (status {response['status']:d}):\n{msg}")

    def get_transcript(self) -> str:
        from queue import Empty
        try:
            ret = self.results.get(timeout=20)
        except Empty:
            raise TimeoutError()
        return ret

    def closed(self, code, reason=None):
        self.results.put(self.trans)


def asr(wav_bytes: bytes, bitrate=2 * 16000):
    content_type = f"audio/x-raw, layout=(string)interleaved, rate=(int){bitrate // 2}, " \
                   "format=(string)S16LE, channels=(int)1"
    ws = ASRClient(
        BytesIO(wav_bytes),
        f'{server_config.asr_api_url}?{urllib.parse.urlencode([("content-type", content_type)])}',
        byterate=bitrate
    )
    ws.connect()
    result = ws.get_transcript()
    result = result.replace('<UNK>', '')
    result = result.replace(' ', '')
    return result


def words_from_candidate_transcript(metadata):
    word = ""
    word_list = []
    word_start_time = 0
    # loop through each character
    for i, token in enumerate(metadata.tokens):
        # append character to word if it's not a space
        if token.text != " ":
            if len(word) == 0:
                # <SIL>
                word_dict = dict()
                word_dict["word"] = "<SIL>"
                word_dict["start_time"] = round(word_start_time, 4)
                word_dict["duration"] = round(token.start_time - word_start_time, 4)
                word_list.append(word_dict)

                # log the start time of the new word
                word_start_time = token.start_time

            word = word + token.text
        # word boundary is either a space or the last character in the array
        if token.text == " " or i == len(metadata.tokens) - 1:
            word_duration = token.start_time - word_start_time

            if word_duration < 0:
                word_duration = 0

            word_dict = dict()
            word = word.encode('utf-8', 'surrogateescape').decode('utf-8')
            word_dict["word"] = word
            word_dict["start_time"] = round(word_start_time, 4)
            word_dict["duration"] = round(word_duration, 4)
            word_list.append(word_dict)

            # reset
            word = ""
            word_start_time = token.start_time  # remember the start time of current silence

    return word_list


def postprocess_metadata(metadata):
    res = [{
        "confidence": transcript.confidence,
        "words": words_from_candidate_transcript(transcript),
    } for transcript in metadata.transcripts]
    return res
