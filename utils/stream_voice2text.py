import pyaudio
import numpy as np
import torch
import tempfile
import os
import soundfile as sf
import warnings
import logging
from collections import deque
from qwen_asr import Qwen3ASRModel

# 彻底屏蔽所有警告信息
warnings.filterwarnings("ignore")

# 屏蔽库的日志信息
logging.getLogger("transformers").setLevel(logging.ERROR)
logging.getLogger("qwen_asr").setLevel(logging.ERROR)
logging.getLogger("pyaudio").setLevel(logging.ERROR)

# 屏蔽特定库的警告
import transformers
transformers.logging.set_verbosity_error()

# 屏蔽ALSA/JACK警告
import os
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['PULSE_PROP'] = 'filter.warn=0'

# 屏蔽PyAudio启动警告
import pyaudio
pyaudio.Stream.__init__ = lambda self, *args, **kwargs: None

def get_text_from_voice(model):
    """
    语音识别主函数 - 替换原有的voice2text功能
    
    功能特性：
    - 自动检测语音开始/结束
    - 支持52种语言自动检测
    - 智能过滤无意义结果
    - 静默运行，无冗余输出
    
    Args:
        model: 已初始化的Qwen3ASRModel实例
        
    Returns:
        str: 识别到的文本内容，如无有效识别返回None
    """
    # 音频参数
    CHUNK_SIZE = 1600  # 100ms音频块
    SAMPLE_RATE = 16000
    SILENCE_THRESHOLD = 1000
    SILENCE_DURATION = 1.5
    
    # 初始化音频流
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE
    )
    
    try:
        print("可以说话了")
        
        # 音频缓冲区（10秒容量）
        audio_buffer = deque(maxlen=SAMPLE_RATE * 10)
        silence_counter = 0
        speech_detected = False
        
        while True:
            # 读取音频数据
            audio_data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_buffer.extend(audio_array)
            
            # 语音活动检测
            audio_energy = np.mean(np.abs(audio_array))
            
            if audio_energy > SILENCE_THRESHOLD:
                # 检测到语音
                silence_counter = 0
                if not speech_detected:
                    speech_detected = True
            else:
                # 静音处理
                silence_counter += CHUNK_SIZE / SAMPLE_RATE
                
                # 检测语音结束
                if speech_detected and silence_counter >= SILENCE_DURATION:
                    # 保存临时文件并识别
                    audio_wav = np.array(audio_buffer, dtype=np.int16)
                    with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                        temp_filename = temp_file.name
                    
                    sf.write(temp_filename, audio_wav, SAMPLE_RATE)
                    results = model.transcribe(audio=temp_filename, language=None)
                    os.unlink(temp_filename)
                    
                    if results and len(results) > 0:
                        final_text = results[0].text.strip()
                        if _is_valid_text(final_text):
                            return final_text
                    
                    # 重置状态
                    audio_buffer.clear()
                    speech_detected = False
                    silence_counter = 0
                    
    except KeyboardInterrupt:
        return None
    finally:
        # 清理资源
        stream.stop_stream()
        stream.close()
        audio.terminate()

def _is_valid_text(text):
    """检查文本是否有效"""
    return (text and 
            len(text) > 2 and
            text not in ['嗯', '啊', '呃', '哦', '哼', '哈', '对', '是'] and
            not text.isspace())

if __name__ == "__main__":
    # 测试代码
    model = Qwen3ASRModel.from_pretrained(
        "resource/models/Qwen3-ASR-0.6B",
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_new_tokens=256,
        attn_implementation="flash_attention_2"
    )
    
    while True:
        text = get_text_from_voice(model)
        if text:
            print(f"识别结果: {text}")