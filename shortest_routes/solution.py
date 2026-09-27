import sys

MOD = 10**9 + 7


def main():
    data = sys.stdin.buffer.read().split()
    n = int(data[0])
    a = [int(x) for x in data[1:1 + n]]

    INF = float('inf')
    dist = [INF] * n
    cnt = [0] * n
    dist[0] = 0
    cnt[0] = 1

    # Edges go only forward, so processing stations in order is a valid BFS order.
    for i in range(n):
        if dist[i] == INF:
            continue
        nd = dist[i] + 1
        ci = cnt[i]
        for j in range(i + 1, min(n, i + a[i] + 1)):
            if nd < dist[j]:
                dist[j] = nd
                cnt[j] = ci
            elif nd == dist[j]:
                cnt[j] = (cnt[j] + ci) % MOD

    print(cnt[n - 1] if dist[n - 1] != INF else -1)


main()
