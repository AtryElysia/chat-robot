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

# 从语音设计参考构建可重用的克隆提示
clone_model = Qwen3TTSModel.from_pretrained(
    "resource\\models\\Qwen3-TTS-12Hz-1.7B-Base",
    device_map="cuda:0",
    dtype=torch.bfloat16,
)

voice_clone_prompt = clone_model.create_voice_clone_prompt(
    ref_audio=(ref_wavs[0], sr),   # or "voice_design_reference.wav"
    ref_text=ref_text,
)

sentences = [
    "No problem! I actually... kinda finished those already? If you want to compare answers or something...",
    "What? No! I mean yes but not like... I just think you're... your titration technique is really precise!",
]

# 为多个单次调用重复使用它
wavs, sr = clone_model.generate_voice_clone(
    text=sentences[0],
    language="English",
    voice_clone_prompt=voice_clone_prompt,
)
sf.write("clone_single_1.wav", wavs[0], sr)

wavs, sr = clone_model.generate_voice_clone(
    text=sentences[1],
    language="English",
    voice_clone_prompt=voice_clone_prompt,
)
sf.write("clone_single_2.wav", wavs[0], sr)

# 或者批量生成一次调用
wavs, sr = clone_model.generate_voice_clone(
    text=sentences,
    language=["English", "English"],
    voice_clone_prompt=voice_clone_prompt,
)
for i, w in enumerate(wavs):
    sf.write(f"clone_batch_{i}.wav", w, sr)