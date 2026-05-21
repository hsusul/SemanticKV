from semantic_kv.cache.models import PromptRequest, SemanticType
from semantic_kv.segmentation.segmenter import PromptSegmenter


def test_segments_chat_roles_and_context():
    request = PromptRequest(
        request_id="r1",
        messages=[
            {"role": "system", "content": "System instructions"},
            {"role": "user", "content": "CONTEXT: retrieved document\n\nUSER: question"},
            {"role": "tool", "content": "tool output"},
        ],
    )
    blocks = PromptSegmenter().segment(request)
    assert [b.semantic_type for b in blocks] == [
        SemanticType.SYSTEM,
        SemanticType.RETRIEVED_CONTEXT,
        SemanticType.USER,
        SemanticType.TOOL_OUTPUT,
    ]


def test_segments_raw_labeled_code():
    request = PromptRequest(request_id="r2", raw_prompt="SYSTEM: hi\nCODE:\n```python\nprint(1)\n```")
    types = [b.semantic_type for b in PromptSegmenter().segment(request)]
    assert SemanticType.SYSTEM in types
    assert SemanticType.CODE in types

