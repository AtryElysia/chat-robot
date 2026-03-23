import pyaudio
import torch
import numpy as np
import tempfile
import os
from collections import deque
from qwen_asr import Qwen3ASRModel
import soundfile as sf

# 初始化ASR模型
model = Qwen3ASRModel.from_pretrained(
    "resource/models/Qwen3-ASR-0.6B",
    dtype=torch.bfloat16,
    device_map="cuda:0",
    max_new_tokens=256,
)

# 音频缓冲区设置
audio_buffer = deque(maxlen=16000 * 5)  # 5秒缓冲区
chunk_size = 1600  # 100ms的音频块

# 设置音频流
audio = pyaudio.PyAudio()
stream = audio.open(
    format=pyaudio.paInt16,
    channels=1,
    rate=16000,
    input=True,
    frames_per_buffer=chunk_size
)

print("🎤 正在监听... (按Ctrl+C停止)")

# 语音活动检测参数
silence_threshold = 500  # 静音阈值
silence_duration = 1.5   # 静音持续时间（秒）
silence_counter = 0

# 文本缓冲区
current_text = ""

try:
    while True:
        # 读取音频块
        audio_data = stream.read(chunk_size)
        
        # 转换为numpy数组
        audio_array = np.frombuffer(audio_data, dtype=np.int16)
        
        # 添加到缓冲区
        audio_buffer.extend(audio_array)
        
        # 检查语音活动
        audio_energy = np.mean(np.abs(audio_array))
        
        if audio_energy > silence_threshold:
            # 有语音活动
            silence_counter = 0
        else:
            # 静音
            silence_counter += chunk_size / 16000  # 转换为秒
        
        # 当检测到静音时进行识别
        if silence_counter >= silence_duration and len(audio_buffer) > 16000:
            # 将缓冲区转换为WAV格式
            audio_wav = np.array(audio_buffer, dtype=np.int16)
            
            try:
                # 创建临时WAV文件
                with tempfile.NamedTemporaryFile(suffix='.wav', delete=False) as temp_file:
                    temp_filename = temp_file.name
                
                # 保存为WAV文件
                sf.write(temp_filename, audio_wav, 16000)
                
                # 使用文件路径进行语音识别
                results = model.transcribe(
                    audio=temp_filename,
                    language="Chinese",  # 自动检测语言
                )
                
                # 删除临时文件
                os.unlink(temp_filename)
                
                if results and len(results) > 0:
                    new_text = results[0].text.strip()
                    if new_text and new_text != current_text:
                        # 繁体转简体（如果需要）
                        import zhconv
                        simplified_text = zhconv.convert(new_text, 'zh-cn')
                        print(f"\n识别结果: {simplified_text}")
                        current_text = simplified_text
                
                # 清空缓冲区
                audio_buffer.clear()
                silence_counter = 0
                
            except Exception as e:
                print(f"识别错误: {e}")
        
        # 显示监听状态
        buffer_duration = len(audio_buffer) / 16000
        status = f"监听中... 缓冲区: {buffer_duration:.1f}s"
        if audio_energy > silence_threshold:
            status += " 🔊"
        else:
            status += " 🔇"
        print(status, end='\r')
            
except KeyboardInterrupt:
    print("\n✅ 停止监听")
finally:
    stream.stop_stream()
    stream.close()
    audio.terminate()