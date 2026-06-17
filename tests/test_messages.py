import json

from miniharness.messages import Conversation, Message, normalize_tool_arguments


def test_conversation_to_openai():
    conversation = Conversation()
    conversation.append(Message(role="user", content="hello"))

    assert conversation.to_openai() == [{"role": "user", "content": "hello"}]


def test_invalid_tool_call_arguments_are_provider_safe():
    conversation = Conversation()
    conversation.append(Message(
        role="assistant",
        content="",
        tool_calls=[
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "write_file",
                    "arguments": '{"path": "x"}}',
                },
            }
        ],
    ))

    exported = conversation.to_openai()
    args = exported[0]["tool_calls"][0]["function"]["arguments"]

    parsed = json.loads(args)
    assert parsed == {"_invalid_arguments": '{"path": "x"}}'}


def test_tool_arguments_must_be_json_object_string():
    assert normalize_tool_arguments({"path": "x"}) == '{"path": "x"}'
    assert normalize_tool_arguments("") == "{}"
    assert normalize_tool_arguments("[1, 2]") == '{"_invalid_arguments": "[1, 2]"}'
