import asyncio
from data_fetcher import fetch_user_data

async def main():
    data = await fetch_user_data([101, 102, 103])
    print("Done:", data)

if __name__ == "__main__":
    asyncio.run(main())
