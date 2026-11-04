"""APScheduler cron jobs."""
from apscheduler.schedulers.asyncio import AsyncIOScheduler

_scheduler: AsyncIOScheduler | None = None


def start_scheduler() -> None:
    global _scheduler
    _scheduler = AsyncIOScheduler(timezone="Asia/Kolkata")
    # _scheduler.add_job(nightly_pick, "cron", hour=2, minute=0)
    _scheduler.start()


def stop_scheduler() -> None:
    if _scheduler:
        _scheduler.shutdown(wait=False)
