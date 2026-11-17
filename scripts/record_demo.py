import asyncio
import os
import re
import time
from playwright.async_api import async_playwright

URL = "http://127.0.0.1:8081/"
MODEL = "/data/wzh/robot/mujoco-studio-lab/model.xml"
OUTDIR = "/data/wzh/robot/mujoco-studio-lab/assets"
VIDEO_TMP = "/tmp/mj_video"
SHOT_BEFORE = OUTDIR + "/shot_before.png"
SHOT_AFTER = OUTDIR + "/shot_after.png"

os.makedirs(OUTDIR, exist_ok=True)
os.makedirs(VIDEO_TMP, exist_ok=True)


def set_rgba(blue_value):
    """Rewrite the sphere rgba blue component in model.xml."""
    with open(MODEL, "r") as f:
        txt = f.read()
    new = re.sub(r'(<geom name="sphere"[^>]*rgba="0\.2 0\.6 )[0-9.]+( 1")',
                 r'\g<1>' + str(blue_value) + r'\g<2>', txt)
    if new == txt:
        # fallback: any rgba with 0.2 0.6 X 1
        new = re.sub(r'(rgba="0\.2 0\.6 )[0-9.]+( 1")',
                     r'\g<1>' + str(blue_value) + r'\g<2>', txt)
    with open(MODEL, "w") as f:
        f.write(new)
    print("set rgba blue ->", blue_value, "changed=", new != txt, flush=True)
    return new != txt


async def main():
    t0 = time.time()
    async with async_playwright() as p:
        browser = await p.chromium.launch(
            headless=True,
            args=["--no-sandbox", "--disable-gpu-sandbox",
                  "--use-gl=angle", "--use-angle=swiftshader",
                  "--enable-unsafe-swiftshader"],
        )
        ctx = await browser.new_context(
            viewport={"width": 1280, "height": 800},
            record_video_dir=VIDEO_TMP,
            record_video_size={"width": 1280, "height": 800},
        )
        page = await ctx.new_page()
        print("opening viewer...", flush=True)
        await page.goto(URL, wait_until="domcontentloaded", timeout=30000)
        await page.wait_for_selector("#canvas", timeout=20000)
        await page.wait_for_timeout(3500)   # let websocket stream + render settle
        print("rendered. title:", await page.title(), flush=True)

        # screenshot BEFORE (sphere blue=0.3 -> greenish)
        await page.screenshot(path=SHOT_BEFORE)
        print("shot before saved", flush=True)

        # drag to rotate the camera view
        box = await page.locator("#canvas").bounding_box()
        print("canvas box:", box, flush=True)
        cx, cy = box["x"] + box["width"] / 2, box["y"] + box["height"] / 2
        await page.mouse.move(cx, cy)
        await page.mouse.down()
        for i in range(1, 13):
            await page.mouse.move(cx + i * 14, cy + i * 5, steps=1)
            await page.wait_for_timeout(40)
        await page.mouse.up()
        print("camera dragged", flush=True)
        await page.wait_for_timeout(1200)

        # trigger hot reload: change rgba blue 0.3 -> 1.0
        set_rgba(1.0)
        print("waiting for hot reload...", flush=True)
        await page.wait_for_timeout(4500)   # page auto-reloads on MJCF change
        await page.screenshot(path=SHOT_AFTER)
        print("shot after saved", flush=True)

        # restore original value so repo file stays at documented default
        set_rgba(0.3)
        await page.wait_for_timeout(1200)

        elapsed = time.time() - t0
        print("elapsed before close: %.2fs" % elapsed, flush=True)
        video = page.video
        await ctx.close()      # finalizes video
        await browser.close()
        if video:
            vpath = await video.path()
            print("VIDEO_PATH=" + str(vpath), flush=True)

    print("DONE total %.2fs" % (time.time() - t0), flush=True)

asyncio.run(main())
