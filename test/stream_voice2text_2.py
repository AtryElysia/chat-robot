import pyaudio
import numpy as np
import torch
import tempfile
import os
import soundfile as sf
import warnings
from qwen_asr import Qwen3ASRModel

# 屏蔽Transformers库的警告信息
warnings.filterwarnings("ignore", message=".*pad_token_id.*eos_token_id.*")
warnings.filterwarnings("ignore", message=".*Setting `pad_token_id` to `eos_token_id`.*")

def auto_stop_streaming_asr():
    """
    Qwen-ASR自动停止流式语音识别
    
    重要说明：
    - vLLM后端只支持HuggingFace Hub模型标识符，不支持本地路径
    - 本地模型只能使用Transformers后端
    - 流式识别功能需要vLLM后端，本地模型无法使用真正的流式识别
    """
    print("🎤 Qwen-ASR语音识别启动中...")
    print("💡 检测到本地模型，使用Transformers后端+VAD方案")
    
    # 由于是本地模型，只能使用Transformers后端
    model = Qwen3ASRModel.from_pretrained(
        "resource/models/Qwen3-ASR-0.6B",
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_new_tokens=256,
        attn_implementation="flash_attention_2"
    )
    print("✅ Transformers后端初始化成功")
    
    # 检查是否支持流式识别
    try:
        # 尝试初始化流式状态（vLLM后端专用）
        state = model.init_streaming_state()
        print("❌ 错误：本地模型不支持流式识别功能")
        print("💡 建议：使用HuggingFace Hub模型以获得流式识别功能")
        return _streaming_recognition_with_transformers(model)
    except Exception as e:
        # 流式识别不可用，使用VAD方案
        print("⚠️ 流式识别不可用，使用VAD+文件识别方案")
        return _streaming_recognition_with_transformers(model)

def _streaming_recognition_with_vllm(model):
    """vLLM后端的真正流式识别实现"""
    CHUNK_SIZE = 1600  # 100ms的音频块
    SAMPLE_RATE = 16000
    SILENCE_THRESHOLD = 1000
    SILENCE_DURATION = 1.5
    
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE
    )
    
    print("🎯 vLLM流式识别已启动（真正的实时识别）")
    print("💡 功能特性：")
    print("   • 真正的实时流式识别")
    print("   • 自动语音开始/结束检测")
    print("   • 支持52种语言自动检测")
    print("-" * 50)
    
    try:
        # 初始化vLLM流式状态
        state = model.init_streaming_state(
            context="",
            language=None,
            chunk_size_sec=0.5,
            unfixed_chunk_num=2,
            unfixed_token_num=10
        )
        
        silence_counter = 0
        speech_detected = False
        last_text = ""
        
        while True:
            audio_data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            
            # 语音活动检测
            audio_energy = np.mean(np.abs(audio_array))
            
            if audio_energy > SILENCE_THRESHOLD:
                silence_counter = 0
                if not speech_detected:
                    print("🔊 检测到语音开始...")
                    speech_detected = True
                
                # vLLM流式识别
                state = model.streaming_transcribe(audio_array, state)
                
                # 实时显示结果
                current_text = state.text.strip()
                if current_text and current_text != last_text:
                    new_text = current_text[len(last_text):]
                    if new_text:
                        print(f"📝 实时识别: {new_text}", end='', flush=True)
                        last_text = current_text
                        
            else:
                silence_counter += CHUNK_SIZE / SAMPLE_RATE
                
                if speech_detected and silence_counter >= SILENCE_DURATION:
                    print("\n🔇 检测到语音结束...")
                    state = model.finish_streaming_transcribe(state)
                    final_text = state.text.strip()
                    
                    if _is_valid_text(final_text):
                        print(f"✅ 最终结果: {final_text}")
                        print("-" * 50)
                        return final_text
                    
                    speech_detected = False
                    silence_counter = 0
                    last_text = ""
                    
    except KeyboardInterrupt:
        print("\n👋 程序已停止")
    finally:
        stream.stop_stream()
        stream.close()
        audio.terminate()

def _streaming_recognition_with_transformers(model):
    """Transformers后端的准流式识别实现（使用VAD+临时文件）"""
    CHUNK_SIZE = 1600
    SAMPLE_RATE = 16000
    SILENCE_THRESHOLD = 1000
    SILENCE_DURATION = 1.5
    
    audio = pyaudio.PyAudio()
    stream = audio.open(
        format=pyaudio.paInt16,
        channels=1,
        rate=SAMPLE_RATE,
        input=True,
        frames_per_buffer=CHUNK_SIZE
    )
    
    print("🎯 Transformers+VAD准流式识别已启动")
    print("💡 功能特性：")
    print("   • VAD自动检测语音开始/结束")
    print("   • 录音完成后识别")
    print("   • 兼容性更好")
    print("-" * 50)
    
    try:
        from collections import deque
        audio_buffer = deque(maxlen=SAMPLE_RATE * 10)  # 10秒缓冲区
        silence_counter = 0
        speech_detected = False
        
        while True:
            audio_data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            audio_array = np.frombuffer(audio_data, dtype=np.int16)
            audio_buffer.extend(audio_array)
            
            # 语音活动检测
            audio_energy = np.mean(np.abs(audio_array))
            
            if audio_energy > SILENCE_THRESHOLD:
                silence_counter = 0
                if not speech_detected:
                    print("🔊 检测到语音开始...")
                    speech_detected = True
            else:
                silence_counter += CHUNK_SIZE / SAMPLE_RATE
                
                if speech_detected and silence_counter >= SILENCE_DURATION:
                    print("\n🔇 检测到语音结束，进行识别...")
                    
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
                            print(f"✅ 识别结果: {final_text}")
                            print("-" * 50)
                            return final_text
                    
                    # 重置状态
                    audio_buffer.clear()
                    speech_detected = False
                    silence_counter = 0
                    
    except KeyboardInterrupt:
        print("\n👋 程序已停止")
    finally:
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
    auto_stop_streaming_asr()