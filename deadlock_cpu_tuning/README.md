# Deadlock CPU usage: what can and can't be changed

## Why only 2 cores are busy

Deadlock runs on Source 2. Each frame goes through two threads that have to
run one after the other:

- the **main / game thread** (simulation, input, animation, networking)
- the **render thread** (turning the scene into GPU commands)

The engine already sends side work (particles, physics, some animation,
culling) to a pool of worker threads. Those are the ~20% cores. The two
80–90% cores are the main and render threads. You can't split those across
more cores from outside the game. That would take changes to the engine's
compiled code, and no config or mod can do that. Editing or injecting into the
game's binaries also risks a VAC ban.

The 40% total CPU number is misleading. A game is CPU-limited when its busiest
thread is maxed out, not when all cores are. So what you see is normal for this
engine.

## Step 1: find out what is actually limiting you

With the Windows Game Bar (Win+G → Performance) or MSI Afterburner/RTSS open,
play a match and check:

| GPU usage | Meaning | What helps |
|---|---|---|
| ~95–99% | **GPU-bound**. The CPU isn't the problem. | Lower resolution/FSR/shadows/AA |
| clearly < 90% while FPS is below your target | **CPU-bound** on the main or render thread | See steps 2–4 |

## Step 2: launch options

Steam → Deadlock → Properties → Launch Options:

```
-high +exec autoexec
```

Optional ones to test one at a time. Keep them only if FPS improves:

- `-vulkan`: the Vulkan renderer. On some systems it has less render-thread
  overhead than DX11. It's the one most likely to help a render-thread
  bottleneck.
- `-threads N`: sets the worker thread count, where N is your number of
  **physical** cores. The engine usually detects this correctly already, so it
  often changes nothing. Remove it if FPS drops.

## Step 3: autoexec.cfg

Copy `autoexec.cfg` from this folder to
`<Steam>\steamapps\common\Deadlock\game\citadel\cfg\`.

Your game build may have more threading options. To list them, open the
console (enable it in Settings → Options → Enable Console, then press F7 or
`~`) and type:

```
find thread
find job
```

Commands differ between patches, so test any you find there rather than
trusting lists online.

## Step 4: Windows-side fixes that help main-thread-bound games

These help more than config tweaks, because they make the two busy threads run
faster:

1. **Power plan**: "High performance" or "Ultimate performance", so busy cores
   aren't parked or clocked down.
2. **Game Mode ON**, and **Hardware-accelerated GPU scheduling ON**
   (Settings → System → Display → Graphics).
3. **Close overlays and background apps**: Discord overlay, browser tabs with
   video, RGB software, recording tools.
4. **RAM**: enable XMP/EXPO in the BIOS. Main-thread-bound games are very
   sensitive to memory speed, and many PCs run RAM at the slow default.
5. **Temperatures**: if the busy cores hit ~95 °C they throttle. Check with
   HWiNFO.
6. **Update the chipset and GPU drivers.** On AMD dual-CCD CPUs
   (7950X3D/9950X3D) also make sure the AMD chipset driver / Game Bar
   recognises the game so it runs on the V-Cache cores.

## Measuring

Before/after each change, play the same situation (e.g. a lane fight in the
Sandbox) and compare average FPS and 1% lows with `cl_showfps 1` or RTSS.
Change one thing at a time.
