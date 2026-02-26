import asyncio


async def crunch_data(data):
    print("Crunching...")
    await asyncio.sleep(1)  # Replaced blocking code
    return data * 2
