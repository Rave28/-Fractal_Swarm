import asyncio
import random
from auto_browse_agent import trigger_v3_ingestion

TARGET_DOMAINS = [
    "https://github.com/trending",
    "https://news.ycombinator.com",
    "https://doc.rust-lang.org/nightly/edition-guide/",
    "https://arxiv.org/list/cs.AI/recent",
]


async def nightcrawler_daemon():
    print("🌙 [NIGHTCRAWLER] Continuous Ingestion Daemon Initialized.")
    print("🌙 [NIGHTCRAWLER] Waiting for idle cycles...")

    while True:
        target = random.choice(TARGET_DOMAINS)
        print(f"\n🌙 [NIGHTCRAWLER] Waking up. Operating on target: {target}")

        try:
            # Trigger the Gemini 3 Auto-Browse Pipeline quietly
            await trigger_v3_ingestion(
                task_prompt="Digest the latest state-of-the-art developments and architectural patterns.",
                target_url=target,
            )
            print("🌙 [NIGHTCRAWLER] Ingestion successful. Entering deep sleep cycle.")
        except Exception as e:
            print(f"🌙 [NIGHTCRAWLER] Navigation exception on {target}: {str(e)}")

        print("🌙 [NIGHTCRAWLER] Suspending operations for 1 hour...")
        # For testing, we mock a 1-hour sleep but we actually only sleep 5 seconds if we were to test it.
        # However, we will use a realistic continuous loop sleep time here.
        await asyncio.sleep(3600)


if __name__ == "__main__":
    try:
        asyncio.run(nightcrawler_daemon())
    except KeyboardInterrupt:
        print("\n🌙 [NIGHTCRAWLER] Daemon terminated by Commander.")
