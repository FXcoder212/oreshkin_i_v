import sys


def main():
    data = sys.stdin.buffer.read().split()
    n = int(data[0])
    ls = data[1:1 + 2 * n:2]
    rs = data[2:2 + 2 * n:2]
    seg = sorted(zip(map(int, rs), map(int, ls)))

    # Greedy by earliest end: put each session on the satellite that became free
    # latest but still not later than l (best fit); reject if neither is free.
    hi = lo = -1  # end times of the two satellites, hi >= lo
    ans = 0
    for r, l in seg:
        if l >= hi:
            hi = r
        elif l >= lo:
            lo = r
        else:
            continue
        ans += 1
        if lo > hi:
            lo, hi = hi, lo
    print(ans)


main()
