import os
import runpy
from datetime import datetime
import requests

TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "").strip()
TELEGRAM_CHAT_ID = os.getenv("TELEGRAM_CHAT_ID", "").strip()


def telegram_enabled() -> bool:
    return bool(TELEGRAM_BOT_TOKEN and TELEGRAM_CHAT_ID)


def send_telegram(text: str) -> None:
    if not telegram_enabled():
        return

    url = f"https://api.telegram.org/bot{TELEGRAM_BOT_TOKEN}/sendMessage"
    requests.post(
        url,
        data={
            "chat_id": TELEGRAM_CHAT_ID,
            "text": text,
            "disable_web_page_preview": True,
        },
        timeout=20,
    )


def main() -> None:
    try:
        runpy.run_path("rs_leader_radar.py", run_name="__main__")
        send_telegram(
            "[RS-RADAR 실행 완료]\n"
            f"실행 시각: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"
        )
    except Exception as e:
        send_telegram(
            "[RS-RADAR 실행 실패]\n"
            f"{type(e).__name__}: {e}"
        )
        raise


if __name__ == "__main__":
    main()
