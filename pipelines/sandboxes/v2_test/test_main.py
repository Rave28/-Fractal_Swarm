import asyncio
from data_fetcher import fetch_user_data

def test_fetch_user_data():
    # Takes 1 second concurrently
    result = asyncio.run(fetch_user_data([1, 2]))
    result.sort(key=lambda x: x["id"])
    assert len(result) == 2
    assert result[0]["id"] == 1
    assert result[1]["id"] == 2
