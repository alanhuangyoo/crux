"""pi clamps a thinking level away unless the model says it can reason.

harbor registers a custom endpoint as `{"id": model_id}` and nothing else. pi
reads `model.reasoning` as false, `getSupportedThinkingLevels` returns
`["off"]`, and every `--thinking` becomes `off` before a request is built --
which is what both arms of the round meant to test the reasoning level recorded
in their own session logs.
"""


class FakeSuper:
    def __init__(self, payload):
        self.payload = payload

    def _build_custom_models_json(self, access, model_id):
        return self.payload


def declare_reasoning(models_json):
    """The body of CruxPiAgent._build_custom_models_json, minus the super() call."""
    if not models_json:
        return models_json
    for provider in models_json.get("providers", {}).values():
        for model in provider.get("models", []):
            model["reasoning"] = True
    return models_json


def harbor_shape(model_id="qwen3.8-27b"):
    return {
        "providers": {
            "harbor-endpoint": {
                "baseUrl": "http://x/v1",
                "apiKey": "$OPENAI_API_KEY",
                "api": "openai-completions",
                "models": [{"id": model_id}],
            }
        }
    }


def test_harbor_alone_leaves_the_model_unable_to_reason():
    entry = harbor_shape()["providers"]["harbor-endpoint"]["models"][0]
    assert "reasoning" not in entry


def test_reasoning_is_declared_on_every_model():
    out = declare_reasoning(harbor_shape())
    for provider in out["providers"].values():
        for model in provider["models"]:
            assert model["reasoning"] is True


def test_nothing_else_is_touched():
    out = declare_reasoning(harbor_shape())
    p = out["providers"]["harbor-endpoint"]
    assert p["baseUrl"] == "http://x/v1"
    assert p["api"] == "openai-completions"
    assert p["models"][0]["id"] == "qwen3.8-27b"


def test_none_passes_through():
    assert declare_reasoning(None) is None
    assert declare_reasoning({}) == {}


def test_multiple_models_and_providers():
    j = harbor_shape()
    j["providers"]["harbor-endpoint"]["models"].append({"id": "other"})
    j["providers"]["second"] = {"models": [{"id": "third"}]}
    out = declare_reasoning(j)
    ids = {m["id"]: m["reasoning"] for p in out["providers"].values() for m in p["models"]}
    assert ids == {"qwen3.8-27b": True, "other": True, "third": True}
