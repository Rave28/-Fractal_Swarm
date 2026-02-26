import asyncio
from processor import crunch_data
from storage import save_to_db


async def run_pipeline(dataset):
    queue = asyncio.Queue()
    results = []

    async def worker():
        while True:
            item = await queue.get()
            if item is None:
                queue.task_done()
                break
            processed = await crunch_data(item)
            await save_to_db(processed)
            results.append(processed)
            queue.task_done()

    async with asyncio.TaskGroup() as tg:
        workers = [tg.create_task(worker()) for _ in range(3)]

        for item in dataset:
            await queue.put(item)

        await queue.join()

        for _ in workers:
            await queue.put(None)

    return results


if __name__ == "__main__":
    print(asyncio.run(run_pipeline([1, 2, 3])))
