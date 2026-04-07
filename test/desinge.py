import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

# 使用VoiceDesign模型创建目标风格的参考音频
design_model = Qwen3TTSModel.from_pretrained(
    "resource\\models\\Qwen3-TTS-12Hz-1.7B-VoiceDesign",
    device_map="cuda:0",
    dtype=torch.bfloat16,
)

# 语音设计参考音频的文本
ref_text = "哥哥你看，这只小猫咪在打滚耶～它的肚子好软好软，我可以摸一下吗？就一下嘛！"
# 语音设计参考音频的指令
ref_instruct = "体现撒娇稚嫩的萝莉女声，音调偏高且起伏明显，营造出粘人、做作又刻意卖萌的听觉效果"
# 生成语音设计参考音频
ref_wavs, sr = design_model.generate_voice_design(
    text=ref_text,
    language="English",
    instruct=ref_instruct
)
sf.write("voice_design_reference.wav", ref_wavs[0], sr)