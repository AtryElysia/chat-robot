import torch
import soundfile as sf
from qwen_tts import Qwen3TTSModel

model_path = "resource\\models\\Qwen3-TTS-12Hz-1.7B-Base"
ref_audio_path = "resource\\voice\\我的眼睛漂亮吗.wav"
ref_text  = "我的眼睛漂亮吗？这可不是美瞳哦，是美少女的魔法！"

model = Qwen3TTSModel.from_pretrained(
    model_path,
    device_map="cuda:0",
    dtype=torch.bfloat16,
)


wavs, sr = model.generate_voice_clone(
    text="我的眼睛漂亮吗？这可不是美瞳哦，是美少女的魔法！",
    language="Chinese",
    ref_audio=ref_audio_path,
    ref_text=ref_text,
)
sf.write("output_voice_clone.wav", wavs[0], sr)