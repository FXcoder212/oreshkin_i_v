#include <bits/stdc++.h>
using namespace std;

int main() {
    int n;
    if (scanf("%d", &n) != 1) return 0;
    vector<pair<int, int>> v(n);  // (r, l)
    for (auto &p : v) if (scanf("%d %d", &p.second, &p.first) != 2) return 0;
    sort(v.begin(), v.end());

    // Greedy by earliest end: put each session on the satellite that became free
    // latest but still not later than l (best fit); reject if neither is free.
    long long hi = -1, lo = -1;  // end times of the two satellites, hi >= lo
    int ans = 0;
    for (auto [r, l] : v) {
        if (l >= hi) { hi = r; ans++; }
        else if (l >= lo) { lo = r; ans++; }
        else continue;
        if (lo > hi) swap(lo, hi);
    }
    printf("%d\n", ans);
    return 0;
}
