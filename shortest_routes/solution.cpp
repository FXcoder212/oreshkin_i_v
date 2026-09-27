#include <bits/stdc++.h>
using namespace std;

int main() {
    const long long MOD = 1000000007LL;
    int n;
    scanf("%d", &n);
    vector<int> a(n);
    for (auto &x : a) scanf("%d", &x);

    const int INF = INT_MAX;
    vector<int> dist(n, INF);
    vector<long long> cnt(n, 0);
    dist[0] = 0;
    cnt[0] = 1;

    // Edges go only forward, so processing stations in order is a valid BFS order.
    for (int i = 0; i < n; i++) {
        if (dist[i] == INF) continue;
        int nd = dist[i] + 1;
        int hi = min(n - 1, i + a[i]);
        for (int j = i + 1; j <= hi; j++) {
            if (nd < dist[j]) { dist[j] = nd; cnt[j] = cnt[i]; }
            else if (nd == dist[j]) { cnt[j] = (cnt[j] + cnt[i]) % MOD; }
        }
    }
    printf("%lld\n", dist[n - 1] == INF ? -1LL : cnt[n - 1]);
    return 0;
}
