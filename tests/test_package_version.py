import flyai_bench


def test_runtime_version_matches_release_version():
    assert flyai_bench.__version__ == "0.1.2"
