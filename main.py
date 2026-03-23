from utils.chat_service import create_chat_service
from utils.text2voice import text2voice
from utils.stream_voice2text import get_text_from_voice
from qwen_tts import Qwen3TTSModel
from qwen_asr import Qwen3ASRModel
import lmstudio as lms
import torch
import re

def remove_brackets(response):

    response = re.sub(r'\【.*?\】', '', response) 
    response = re.sub(r'\[.*?\]', '', response) 
    response = re.sub(r'\(.*?\)', '', response) 
    response = response.strip() 
    return response

def load_model():

    Qwen3_TTS_12Hz_17B_Base = Qwen3TTSModel.from_pretrained(
        "resource/models/Qwen3-TTS-12Hz-1.7B-Base",
        device_map="cuda:0",
        dtype=torch.bfloat16,
        # attn_implementation="flash_attention_2",
    )

    Qwen3_ASR_0_6B = Qwen3ASRModel.from_pretrained(
        "resource/models/Qwen3-ASR-0.6B",
        dtype=torch.bfloat16,
        device_map="cuda:0",
        max_new_tokens=256,
        # attn_implementation="flash_attention_2",
    )

    Qwen3_4B_Instruct_2507_Nekoqa_10k_Unloth_Bnb_Finetune = lms.llm('qwen3-4b-instruct-2507-nekoqa-10k-unsloth-bnb-finetune')
    
    return Qwen3_TTS_12Hz_17B_Base, Qwen3_ASR_0_6B, Qwen3_4B_Instruct_2507_Nekoqa_10k_Unloth_Bnb_Finetune

def main(Qwen3_ASR_0_6B, Qwen3_4B_Instruct_2507_Nekoqa_10k_Unloth_Bnb_Finetune, Qwen3_TTS_12Hz_17B_Base):
        
    def get_voice_input():
        return get_text_from_voice(Qwen3_ASR_0_6B)
    
    def generate_response(user_input):
        response = create_chat_service(Qwen3_4B_Instruct_2507_Nekoqa_10k_Unloth_Bnb_Finetune, user_input)
        return remove_brackets(response)
    
    # 获取用户输入
    user_input = get_voice_input()
    if not user_input:
        return
        
    print("你: " + user_input)
    
    # 生成响应
    response = generate_response(user_input)
    print("猫娘: " + response)
    
    # 语音合成
    text2voice(Qwen3_TTS_12Hz_17B_Base, response)

if __name__ == "__main__":
    # 预加载所有模型（只加载一次）
    print("正在加载模型，请稍候...")
    Qwen3_TTS_12Hz_17B_Base, Qwen3_ASR_0_6B, Qwen3_4B_Instruct_2507_Nekoqa_10k_Unloth_Bnb_Finetune = load_model()
    print("模型加载完成！开始对话...")
    
    while True:
        main(Qwen3_ASR_0_6B, Qwen3_4B_Instruct_2507_Nekoqa_10k_Unloth_Bnb_Finetune, Qwen3_TTS_12Hz_17B_Base)