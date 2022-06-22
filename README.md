Public version of a Computer-Aided Pronunciation Training backend I made as my undergraduate signature work

The client is located [here](https://github.com/tjysdsg/hippo)

# Important Notes

The project was in a rush, so many lines of dead/bad code and unused files

## Things obfuscated or removed

- IP addresses of external ASR and TTS servers in `configs/external_api.json`
- The remote database's URL in `gop_server/db.py`
- Lists of dialogs and their corresponding reference audio in `gop_chinese/word_list`
- [ToneNet](https://www.isca-speech.org/archive_v0/Interspeech_2019/pdfs/1483.pdf) model that I trained (`ToneNet.hdf5`
  and `ToneNet.pkl`) in `gop_chinese`

### Kaldi models

The alignment and scoring of speech input is two separate pass because we want the alignment to be error-insensitive,
while scoring be error-sensitive.

- `tdnn_align/` contains a Kaldi TDNN model (not include in the repo) used for aligning speech input
- `tdnn_score/` contains a Kaldi TDNN model (not include in the repo) used for GOP scoring

The file tree is saved as `tree.txt` in both folders for your reference.

The alignment model was trained with AISHELL-2, while the scoring model is trained
like [this](https://github.com/tjysdsg/std-mandarin-kaldi)

## Design Notes

The algorithm is implemented using a combination of Cython and PyKaldi.
This is a very bad call.
PyKaldi does not seem to be actively maintained.
And it makes environment setup difficult.
PyBind+Kaldi could have been great, easy integration with the newest version of Kaldi and the speed should be much
better.

Many things are hardcoded.
A lot of code can be refactored and optimized.
The web API design is awful.
Therefore, the only value of this repo for you is probably the implementation of GOP and its integration to a web
server.

# Dependencies

See `scripts/install-deps.sh`

# Core API Documentation

## Compute GOP of Arbitrary Text

- Address: `/gop`
- Interface type: websocket. The user must first upload metadata and then upload wav bytes using this interface.
- Input 1: metadata. Its json format is:

```json5
{
  // extension is of 'webm', 'wav', 'm4a', and 'caf'
  "ext": "string",
  "transcript": "你好",
}
```

- Input 2: audio data (binary type).
- Example output:

```json5
{
  // should be 0 if successful
  'status': 0,
  // error message if status != 0
  'message': '',
  'data': {
    'pinyin': [
      // pinyin
      'n',
      'í',
      'h',
      'ǎo'
    ],
    'gop': [
      // GOPs of each phone, not recommended to be used directly (since the scale of shengmu and yunmu GOPs is different)
      // NOTE that this is independent from tone,
      //   meaning the value of "corr" is true as long as pronunciation of a phone is correct (even if the tone is incorrect)
      0.0,
      0.12682069858108927,
      0.0,
      2.633087972147811,
    ],
    // correctness of phone pronunciation
    // NOTE that this is independent from tone,
    //   meaning the value of "corr" is true as long as pronunciation of a phone is correct (even if the tone is incorrect)
    'corr': [
      true,
      true,
      true,
      false
    ],
    // perceived tone index of each phone, within [0, 4]
    // tone of a shengmu is none, represented by 0
    // otherwise 1 to 4 represents 一声 to 三声
    'tones': [
      0,
      3,
      0,
      2
    ]
  }
}
```

## Internal web APIs (beteen client and server)