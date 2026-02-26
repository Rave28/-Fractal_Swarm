import asyncio
from main import run_pipeline


def test_run():
    # Because of unordered queue processing, sort results for assertion
    results = asyncio.run(run_pipeline([10, 20]))
    assert sorted(results) == [20, 40]
