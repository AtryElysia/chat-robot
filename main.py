from utils.chat_service import create_chat_service
from utils.text2voice import text2voice
from utils.stream_voice2text import get_text_from_voice
from qwen_tts import Qwen3TTSModel
from qwen_asr import Qwen3ASRModel
import lmstudio as lms
import torch
import re
import json
import os

def remove_brackets(response):
    response = re.sub(r'\【.*?\】', '', response) 
    response = re.sub(r'\[.*?\]', '', response) 
    response = re.sub(r'\(.*?\)', '', response) 
    response = response.strip() 
    return response

def get_wake_config(config):
    """从配置中获取唤醒词相关配置"""
    return {
        "wake_words": config.get("wake_words", ["猫娘", "小助手", "助手", "你好", "嗨", "嘿"]),
        "system_prompt": config.get("system_prompt", "你是一只猫娘，每句话末尾都要加个喵。请用可爱、活泼的语气回答用户的问题。")
    }

def contains_wake_word(text, wake_words):
    """检测文本中是否包含唤醒词"""
    if not text:
        return False
    
    text_lower = text.lower()
    for word in wake_words:
        if word.lower() in text_lower:
            return True
    return False

def get_wake_word_response(text, wake_words):
    """处理包含唤醒词的文本，返回适当的响应"""
    if not text:
        return text
    
    # 找到第一个唤醒词的位置
    text_lower = text.lower()
    first_wake_pos = -1
    first_wake_word = ""
    
    for word in wake_words:
        pos = text_lower.find(word.lower())
        if pos != -1 and (first_wake_pos == -1 or pos < first_wake_pos):
            first_wake_pos = pos
            first_wake_word = word
    
    if first_wake_pos != -1:
        # 提取唤醒词之后的内容
        start_pos = first_wake_pos + len(first_wake_word)
        user_message = text[start_pos:].strip()
        
        # 如果用户消息为空，可能是只有唤醒词
        if not user_message:
            return "请告诉我你需要什么帮助？"
        
        # 返回完整的原始文本，让大模型自己处理
        return text
    
    return text

def load_config():
    """加载配置文件"""
    config_path = "config.json"
    if not os.path.exists(config_path):
        # 如果配置文件不存在，创建默认配置
        default_config = {
            "models": {
                "tts_model": "resource/models/Qwen3-TTS-12Hz-1.7B-Base",
                "asr_model": "resource/models/Qwen3-ASR-0.6B",
                "llm_model": "qwen3-4b-instruct-2507-nekoqa-10k-unsloth-bnb-finetune"
            },
            "device": "cuda:0",
            "dtype": "bfloat16",
            "asr_max_new_tokens": 256,
            "wake_words": [
                "猫娘",
                "小助手",
                "助手", 
                "你好",
                "嗨",
                "嘿"
            ],
            "system_prompt": "你是一只猫娘，每句话末尾都要加个喵。请用可爱、活泼的语气回答用户的问题。"
        }
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(default_config, f, ensure_ascii=False, indent=2)
        print(f"已创建默认配置文件: {config_path}")
        return default_config
    
    try:
        with open(config_path, 'r', encoding='utf-8') as f:
            config = json.load(f)
        return config
    except Exception as e:
        print(f"加载配置文件失败: {e}")
        # 返回默认配置
        return {
            "models": {
                "tts_model": "resource/models/Qwen3-TTS-12Hz-1.7B-Base",
                "asr_model": "resource/models/Qwen3-ASR-0.6B",
                "llm_model": "qwen3-4b-instruct-2507-nekoqa-10k-unsloth-bnb-finetune"
            },
            "device": "cuda:0",
            "dtype": "bfloat16",
            "asr_max_new_tokens": 256,
            "wake_words": [
                "猫娘",
                "小助手",
                "助手", 
                "你好",
                "嗨",
                "嘿"
            ],
            "system_prompt": "你是一只猫娘，每句话末尾都要加个喵。请用可爱、活泼的语气回答用户的问题。"
        }

def load_model(config):
    """根据配置加载模型"""
    models_config = config.get("models", {})
    device = config.get("device", "cuda:0")
    dtype_str = config.get("dtype", "bfloat16")
    asr_max_new_tokens = config.get("asr_max_new_tokens", 256)
    
    # 转换dtype字符串为torch.dtype
    if dtype_str.lower() in ["bfloat16", "bf16"]:
        dtype = torch.bfloat16
    elif dtype_str.lower() in ["float16", "fp16"]:
        dtype = torch.float16
    elif dtype_str.lower() in ["float32", "fp32"]:
        dtype = torch.float32
    else:
        dtype = torch.bfloat16
        print(f"警告: 未知的dtype '{dtype_str}'，使用默认值bfloat16")
    
    # 加载TTS模型
    tts_model_path = models_config.get("tts_model", "resource/models/Qwen3-TTS-12Hz-1.7B-Base")
    print(f"正在加载TTS模型: {tts_model_path}")
    Qwen3_TTS_12Hz_17B_Base = Qwen3TTSModel.from_pretrained(
        tts_model_path,
        device_map=device,
        dtype=dtype,
        # attn_implementation="flash_attention_2",
    )
    print("✅ TTS模型加载完成")

    # 加载ASR模型
    asr_model_path = models_config.get("asr_model", "resource/models/Qwen3-ASR-0.6B")
    print(f"正在加载ASR模型: {asr_model_path}")
    Qwen3_ASR_0_6B = Qwen3ASRModel.from_pretrained(
        asr_model_path,
        dtype=dtype,
        device_map=device,
        max_new_tokens=asr_max_new_tokens,
        # attn_implementation="flash_attention_2",
    )
    print("✅ ASR模型加载完成")

    # 加载LLM模型
    llm_model_name = models_config.get("llm_model", "qwen3-4b-instruct-2507-nekoqa-10k-unsloth-bnb-finetune")
    print(f"正在加载LLM模型: {llm_model_name}")
    Qwen3_4B_Instruct_2507_Nekoqa_10k_Unloth_Bnb_Finetune = lms.llm(llm_model_name)
    print("✅ LLM模型加载完成")
    
    return Qwen3_TTS_12Hz_17B_Base, Qwen3_ASR_0_6B, Qwen3_4B_Instruct_2507_Nekoqa_10k_Unloth_Bnb_Finetune

def main(Qwen3_ASR_0_6B, Qwen3_4B_Instruct_2507_Nekoqa_10k_Unloth_Bnb_Finetune, Qwen3_TTS_12Hz_17B_Base, wake_config):
        
    def get_voice_input():
        return get_text_from_voice(Qwen3_ASR_0_6B)
    
    def generate_response(user_input, system_prompt=None):
        response = create_chat_service(Qwen3_4B_Instruct_2507_Nekoqa_10k_Unloth_Bnb_Finetune, user_input, system_prompt)
        return remove_brackets(response)
    
    # 获取用户输入
    user_input = get_voice_input()
    if not user_input:
        return
    
    print(f"识别结果: {user_input}")
    
    # 检测是否包含唤醒词
    wake_words = wake_config.get("wake_words", [])
    system_prompt = wake_config.get("system_prompt", "")
    
    if contains_wake_word(user_input, wake_words):
        print("🔔 检测到唤醒词，开始对话...")
        
        # 保留完整的用户输入，让大模型自己处理唤醒词
        user_message = get_wake_word_response(user_input, wake_words)
        print(f"你: {user_input}")
        
        # 生成响应
        response = generate_response(user_message, system_prompt)
        print(f"猫娘: {response}")
        
        # 语音合成
        text2voice(Qwen3_TTS_12Hz_17B_Base, response)
    else:
        print("❌ 未检测到唤醒词，忽略本次输入")
        print("💡 支持的唤醒词:", ", ".join(wake_words))

if __name__ == "__main__":
    # 加载配置文件
    config = load_config()
    
    # 获取唤醒词配置
    wake_config = get_wake_config(config)
    wake_words = wake_config.get("wake_words", [])
    
    print("🔊 唤醒词检测功能已启用")
    print("📋 支持的唤醒词:", ", ".join(wake_words))
    print("⚙️  当前配置:")
    print(f"   设备: {config.get('device', 'cuda:0')}")
    print(f"   数据类型: {config.get('dtype', 'bfloat16')}")
    print("-" * 50)
    
    # 预加载所有模型（只加载一次）
    print("正在加载模型，请稍候...")
    Qwen3_TTS_12Hz_17B_Base, Qwen3_ASR_0_6B, Qwen3_4B_Instruct_2507_Nekoqa_10k_Unloth_Bnb_Finetune = load_model(config)
    print("模型加载完成！开始对话...")
    print("💡 请说出包含唤醒词的指令来开始对话")
    print("-" * 50)
    
    while True:
        main(Qwen3_ASR_0_6B, Qwen3_4B_Instruct_2507_Nekoqa_10k_Unloth_Bnb_Finetune, Qwen3_TTS_12Hz_17B_Base, wake_config)