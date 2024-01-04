import torch
from TTS.api import TTS

# Get device
device = "cuda" if torch.cuda.is_available() else "cpu"

# List available 🐸TTS models
print(TTS().list_models())
print(device)


# Init TTS

model_mult = "tts_models/multilingual/multi-dataset/xtts_v2"

model_zh = "tts_models/zh-CN/baker/tacotron2-DDC-GST"


tts = TTS("tts_models/multilingual/multi-dataset/xtts_v2").to(device)

#tts = TTS(model_zh).to(device)





t1 = '''

见鬼，神像还没有好，没法加血

哈哈哈，需要加血吗？我可以给你加点血

但是我不想给你加

听着，兄弟，你没必要干掉我，这里好多宝箱，你拿你的，我拿我的，我们都能逃出去，行吗？

兄弟，你也太天真了



'''


wav = tts.tts_to_file(text=t1,speaker_wav="./sample/test_dark.WAV",language="zh-cn",file_path="output.wav")

#wav = tts.tts_to_file(text=t1,file_path="output_cn.wav")

# Run TTS
# ❗ Since this model is multi-lingual voice cloning model, we must set the target speaker_wav and language
# Text to speech list of amplitude values as output
# wav = tts.tts_to_file(text=t1, speaker_wav="./test_speech.wav", language="zh-cn",file_path="output.wav")
