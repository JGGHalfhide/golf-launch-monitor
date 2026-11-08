"""
USB clip source: watch a tethered iPhone's camera roll over AFC and download new videos.

AFC (Apple File Conduit) reads the phone's DCIM folders directly. Unlike ImageCaptureCore/PTP
(which, with "Optimize iPhone Storage" on, only exposes a small stale subset of the library and
never showed new clips), AFC sees a clip as soon as the Camera app saves it.

Runs in a background thread. Every poll it lists the newest DCIM folders, and each original
video (IMG_1234.MOV) that wasn't there at connect time is downloaded once its size stops
changing — to <inbox>/.incoming/ first, then renamed into <inbox>/ so the consumer never sees
a partial file. On (re)connect the current contents are the baseline: anything recorded while
the watcher wasn't connected is ignored, so a late clip can never be sent out of order.
"""
import asyncio
import re
import threading
import time
from pathlib import Path
from typing import Callable

ORIGINAL_VIDEO = re.compile(r"^IMG_\d{4}\.(MOV|MP4)$", re.IGNORECASE)  # not IMG_E… (edited renders)
DCIM_FOLDER = re.compile(r"^\d{3}APPLE$")
FOLDERS_TO_WATCH = 2  # newest folders; covers rollover from e.g. 140APPLE → 141APPLE


def is_original_video(name: str) -> bool:
    return bool(ORIGINAL_VIDEO.match(name))


def newest_folders(entries: list[str], n: int = FOLDERS_TO_WATCH) -> list[str]:
    return sorted(e for e in entries if DCIM_FOLDER.match(e))[-n:]


def stable_new_files(
    sizes: dict[str, int],
    prev_sizes: dict[str, int],
    known: set[str],
) -> list[str]:
    """Paths not yet known whose size is non-zero and unchanged since the previous poll."""
    return sorted(p for p, sz in sizes.items()
                  if p not in known and sz > 0 and prev_sizes.get(p) == sz)


class AfcSource:
    """
    on_event(kind, **fields) is called from the watcher thread with kinds:
      connected, disconnected, waiting (no phone yet), downloading, downloaded, error.
    """

    def __init__(self, inbox: Path, on_event: Callable[..., None], poll_s: float = 0.75):
        self.inbox = inbox
        self.incoming = inbox / ".incoming"
        self.on_event = on_event
        self.poll_s = poll_s
        self._stop = threading.Event()
        self.thread = threading.Thread(target=lambda: asyncio.run(self._run()), daemon=True)

    def start(self) -> None:
        self.incoming.mkdir(parents=True, exist_ok=True)
        self.thread.start()

    def stop(self) -> None:
        self._stop.set()

    def is_alive(self) -> bool:
        return self.thread.is_alive()

    async def _run(self) -> None:
        from pymobiledevice3.lockdown import create_using_usbmux
        from pymobiledevice3.services.afc import AfcService

        announced_waiting = False
        while not self._stop.is_set():
            try:
                lockdown = await create_using_usbmux()
                afc = AfcService(lockdown)
            except Exception as e:
                if not announced_waiting:
                    self.on_event("waiting", message=str(e) or type(e).__name__)
                    announced_waiting = True
                await asyncio.sleep(2)
                continue
            announced_waiting = False
            try:
                await self._watch(afc, getattr(lockdown, "display_name", None) or "iPhone")
            except Exception as e:
                self.on_event("disconnected", message=str(e) or type(e).__name__)
                await asyncio.sleep(2)

    async def _list_videos(self, afc) -> dict[str, int]:
        sizes: dict[str, int] = {}
        for folder in newest_folders(await afc.listdir("/DCIM")):
            for name in await afc.listdir(f"/DCIM/{folder}"):
                if is_original_video(name):
                    path = f"/DCIM/{folder}/{name}"
                    sizes[path] = int((await afc.stat(path))["st_size"])
        return sizes

    async def _watch(self, afc, device_name: str) -> None:
        known = set(await self._list_videos(afc))  # baseline: everything already on the phone
        self.on_event("connected", name=device_name, existing=len(known))
        prev: dict[str, int] = {}
        while not self._stop.is_set():
            sizes = await self._list_videos(afc)
            for path in stable_new_files(sizes, prev, known):
                known.add(path)
                await self._download(afc, path, sizes[path])
            prev = sizes
            await asyncio.sleep(self.poll_s)

    async def _download(self, afc, path: str, size: int) -> None:
        name = path.rsplit("/", 1)[-1]
        self.on_event("downloading", name=name, bytes=size)
        t0 = time.time()
        try:
            await afc.pull(path, str(self.incoming), progress_bar=False)
            src = self.incoming / name
            dst = self.inbox / f"{int(time.time())}-{name}"
            src.replace(dst)
        except Exception as e:
            self.on_event("error", stage="download", name=name, message=str(e) or type(e).__name__)
            return
        self.on_event("downloaded", name=name, path=str(dst), seconds=time.time() - t0)
