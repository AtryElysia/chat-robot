import torch
import soundfile as sf
import numpy as np
from qwen_tts import Qwen3TTSModel
import pyaudio
import warnings
import logging

# 屏蔽警告信息
warnings.filterwarnings("ignore")
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("qwen_tts").setLevel(logging.ERROR)

def text2voice(model, text):
    """
    文本转语音，支持CPU回退机制
    """
    ref_audio_path = "resource/voice/我的眼睛漂亮吗.wav"
    ref_text  = "我的眼睛漂亮吗？这可不是美瞳哦，是美少女的魔法！"

    try:
        # 尝试使用GPU生成
        wavs, sr = model.generate_voice_clone(
            text=text,
            language="Chinese",
            ref_audio=ref_audio_path,
            ref_text=ref_text,
        )
    except Exception as e:
        # GPU失败，回退到CPU
        print(f"GPU生成失败，切换到CPU模式: {e}")
        
        # 将模型移动到CPU
        model = model.cpu()
        torch.cuda.empty_cache()
        
        wavs, sr = model.generate_voice_clone(
            text=text,
            language="Chinese",
            ref_audio=ref_audio_path,
            ref_text=ref_text,
        )
    
    # 检查音频数据是否有效
    if len(wavs) == 0 or wavs[0] is None:
        raise ValueError("生成的音频数据为空")
    
    # 将浮点音频数据转换为16位整数格式
    audio_data = wavs[0]
    audio_data_int16 = np.int16(audio_data * 32767)
    
    # 使用pyaudio播放音频（修复警告）
    p = pyaudio.PyAudio()
    
    # 获取可用的音频设备
    device_info = None
    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)
        if info['maxOutputChannels'] > 0:
            device_info = info
            break
    
    # 使用默认设备或第一个可用设备
    if device_info:
        device_index = device_info['index']
    else:
        device_index = None
    
    stream = p.open(format=pyaudio.paInt16, 
                       channels=1,
                       rate=int(sr), 
                       output=True)
        
        # 播放音频
    stream.write(audio_data_int16.tobytes())
        
    """
    # 等待播放完成
    duration = len(audio_data_int16) / sr
    import time
    time.sleep(duration)
    """