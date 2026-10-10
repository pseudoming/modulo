import subprocess
import sys


def test_tokenizer_only_startup_quiet_but_other_warnings_visible():
    # A fresh process exercises the real first import rather than the already
    # cached Transformers module from other tests.
    code = (
        "import logging\n"
        "from modulo.adapters.tokenizers.shared.huggingface import _auto_tokenizer\n"
        "print(_auto_tokenizer().__name__)\n"
        "logging.getLogger('transformers').warning('visible-diagnostic-probe')\n"
    )
    completed = subprocess.run(
        [sys.executable, "-c", code], capture_output=True, text=True, timeout=30, check=True
    )
    assert completed.stdout.strip() == "AutoTokenizer"
    assert "None of PyTorch" not in completed.stderr
    assert "visible-diagnostic-probe" in completed.stderr
