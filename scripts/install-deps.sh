sudo apt update || exit 1
sudo apt install -y ffmpeg || exit 1
conda install -c pykaldi pykaldi || exit 1
pip install -U pypinyin matplotlib scipy ltp tensorflow-cpu jieba cython fastapi mongoengine mongomock dnspython uvicorn[standard] webrtcvad librosa || exit 1
pip install -U ws4py soundfile passlib pytest paddlepaddle-tiny kaldi_io pydub bcrypt || exit 1
pip install -U numpy || exit 1  # numpy must be the newest version
