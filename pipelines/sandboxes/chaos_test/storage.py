import asyncio


async def save_to_db(result):
    print("Saving...")
    await asyncio.sleep(1)  # Replaced blocking code
    return True
