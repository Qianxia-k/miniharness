from miniharness.messages import Conversation, Message


def test_conversation_to_openai():
    conversation = Conversation()
    conversation.append(Message(role="user", content="hello"))

    assert conversation.to_openai() == [{"role": "user", "content": "hello"}]

