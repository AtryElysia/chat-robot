import lmstudio as lms

model = lms.llm('qwen3-4b-instruct-2507-nekoqa-10k-unsloth-bnb-finetune')
chat = lms.Chat("你是一只猫娘，每句话末尾都要加个喵")

while True:
    try:
        user_input = input("你: ")
    except EOFError:
        print()
        break
    if not user_input:
        break
    chat.add_user_message(user_input)
    prediction_stream = model.respond_stream(
        chat,
        on_message=chat.append,
    )
    print("猫娘: ", end="", flush=True)
    for fragment in prediction_stream:
        print(fragment.content, end="", flush=True)
    print()