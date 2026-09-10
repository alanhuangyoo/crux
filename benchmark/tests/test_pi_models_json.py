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
            compat = model.setdefault("compat", {})
            compat.setdefault("supportsDeveloperRole", False)
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


def test_the_role_stays_system():
    """`reasoning: true` alone makes pi send `role: "developer"`, which this
    server answers with `{"message":"Unexpected message role."}` and a 400 --
    every turn, so a smoke run died at turn one and scored three ordinary
    zeros."""
    out = declare_reasoning(harbor_shape())
    model = out["providers"]["harbor-endpoint"]["models"][0]
    assert model["compat"]["supportsDeveloperRole"] is False


def test_an_explicit_compat_is_not_overwritten():
    j = harbor_shape()
    j["providers"]["harbor-endpoint"]["models"][0]["compat"] = {
        "supportsDeveloperRole": True,
        "supportsStore": False,
    }
    out = declare_reasoning(j)
    compat = out["providers"]["harbor-endpoint"]["models"][0]["compat"]
    assert compat["supportsDeveloperRole"] is True
    assert compat["supportsStore"] is False


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
    for p in out["providers"].values():
        for m in p["models"]:
            assert m["compat"]["supportsDeveloperRole"] is False


# --- turning reasoning off at the chat template ---


def declare_reasoning_no_thinking(models_json):
    """`declare_reasoning`, with the opt-in switch on."""
    out = declare_reasoning(models_json)
    for provider in out["providers"].values():
        for model in provider["models"]:
            compat = model["compat"]
            compat["thinkingFormat"] = "chat-template"
            kwargs = compat.setdefault("chatTemplateKwargs", {})
            kwargs.setdefault("enable_thinking", False)
            kwargs.setdefault("preserve_thinking", True)
    return out


def test_off_by_default():
    model = declare_reasoning(harbor_shape())["providers"]["harbor-endpoint"]["models"][0]
    assert "chatTemplateKwargs" not in model["compat"]
    assert "thinkingFormat" not in model["compat"]


def test_the_switch_reaches_the_chat_template():
    # `reasoning_effort` biases this model rather than capping it: with
    # effort=low the median thinking block on the five tasks reasoning kills is
    # still 41,888-54,778 characters. The chat template is the control the
    # server honours absolutely -- 0, 0, 0 characters across three samples.
    model = declare_reasoning_no_thinking(harbor_shape())["providers"]["harbor-endpoint"]["models"][0]
    assert model["compat"]["chatTemplateKwargs"]["enable_thinking"] is False
    assert model["reasoning"] is True
    assert model["compat"]["supportsDeveloperRole"] is False


def test_the_format_is_the_one_this_server_honours():
    # Two samples each against this server, characters of reasoning_content:
    #   chat_template_kwargs{enable_thinking:0}      0      0
    #   chat_template_args{enable_thinking:0}     9211   9046   <- pi "baseten"
    #   top-level enable_thinking:false           9011   8828   <- pi "qwen"
    # The first version of this shipped the baseten field and the probe ran
    # with reasoning fully on.
    compat = declare_reasoning_no_thinking(harbor_shape())["providers"]["harbor-endpoint"]["models"][0]["compat"]
    assert compat["thinkingFormat"] == "chat-template"
    assert "chatTemplateArgs" not in compat, "baseten field: this server ignores it"
    assert compat["chatTemplateKwargs"]["preserve_thinking"] is True


# --- telling pi how much context the server actually serves ---


def apply_window(models_json, window):
    """The window half of CruxPiAgent._build_custom_models_json."""
    out = declare_reasoning(models_json)
    if not window:
        return out
    for provider in out["providers"].values():
        for model in provider["models"]:
            model["contextWindow"] = window
            model.setdefault("maxTokens", min(16384, window // 2))
    return out


def test_without_a_window_pi_is_left_guessing():
    model = declare_reasoning(harbor_shape())["providers"]["harbor-endpoint"]["models"][0]
    assert "contextWindow" not in model
    assert "maxTokens" not in model


def test_a_small_window_bounds_both_the_context_and_the_ceiling():
    # 32899 = 16515 input + 16384 completion, against a limit of 32768. Without
    # the window pi compacts too late *and* escalates the ceiling into space
    # that is not there; both arrive as the same opaque 400.
    model = apply_window(harbor_shape(), 32768)["providers"]["harbor-endpoint"]["models"][0]
    assert model["contextWindow"] == 32768
    assert model["maxTokens"] == 16384
    assert model["maxTokens"] * 2 <= model["contextWindow"]


def test_a_tiny_window_gets_a_proportionate_ceiling():
    model = apply_window(harbor_shape(), 8192)["providers"]["harbor-endpoint"]["models"][0]
    assert model["maxTokens"] == 4096


def test_an_unreadable_endpoint_changes_nothing():
    # An endpoint that will not answer is not a reason to fail a trial.
    model = apply_window(harbor_shape(), None)["providers"]["harbor-endpoint"]["models"][0]
    assert "contextWindow" not in model
    assert model["reasoning"] is True
