import lmstudio as lms


def create_chat_service(model, user_input):
    chat = lms.Chat("你是一只猫娘")

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