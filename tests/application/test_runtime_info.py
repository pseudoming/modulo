from modulo.application.runtime_info import collect_runtime_info


def test_runtime_info_covers_all_layers_after_relocation():
    info = collect_runtime_info([])
    assert {
        "core/main.py",
        "core/processing/recalculation.py",
        "core/domain/data_models.py",
        "core/domain/platforms.py",
        "core/domain/interfaces.py",
        "shared/serialization.py",
        "application/importer.py",
        "application/registry.py",
        "application/runtime_info.py",
        "application/tokenizer_assets.py",
        "adapters/sources/elasticsearch.py",
        "adapters/protocols/dify/adapter.py",
        "adapters/protocols/dify/agent.py",
        "adapters/protocols/shared/chat.py",
        "adapters/tokenizers/deepseek.py",
        "adapters/tokenizers/shared/huggingface.py",
        "adapters/tokenizers/shared/rendering.py",
        "adapters/tokenizers/shared/assets.py",
        "storage/result_bundle.py",
        "storage/query_index.py",
        "shared/file_hash.py",
        "cli/commands.py",
        "cli/configuration.py",
    } <= info["code"].keys()
    assert "core/pipeline.py" not in info["code"]
    assert "pipeline.py" not in info["code"]
