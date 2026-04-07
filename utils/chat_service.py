import lmstudio as lms


def create_chat_service(model, user_input, system_prompt=None):
    # 使用传入的系统提示词，如果没有则使用默认提示词
    if system_prompt is None:
        system_prompt = "你是一只猫娘"
    
    chat = lms.Chat(system_prompt)

    while True:
        if not user_input:
            break
        chat.add_user_message(user_input)
        prediction_stream = model.respond_stream(
            chat,
            on_message=chat.append,
        )
        # 收集所有响应片段
        full_response = ""
        for fragment in prediction_stream:
            full_response += fragment.content
        
        # 返回完整的响应文本
        return full_response