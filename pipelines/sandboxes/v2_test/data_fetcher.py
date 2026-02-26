import asyncio

async def fetch_user(uid):
    print(f"Fetching {uid}...")
    await asyncio.sleep(1)
    return {"id": uid, "data": "dummy_data"}

async def fetch_user_data(user_ids):
    async with asyncio.TaskGroup() as tg:
        tasks = [tg.create_task(fetch_user(uid)) for uid in user_ids]
    return [task.result() for task in tasks]
