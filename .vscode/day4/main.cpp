
#include <bits/stdc++.h>
using namespace std;

int main() {
    const long long iterations = 200'000'000LL;
    const double param1 = 4.0;
    const double param2 = 1.0;

    auto t0 = chrono::high_resolution_clock::now();

    double result = 1.0;
    for (long long i = 1; i <= iterations; ++i) {
        double i_d = static_cast<double>(i);
        double j = i_d * param1 - param2;
        result -= 1.0 / j;
        j = i_d * param1 + param2;
        result += 1.0 / j;
    }
    result *= 4.0;

    auto t1 = chrono::high_resolution_clock::now();
    chrono::duration<double> elapsed = t1 - t0;

    printf("Result: %.12f\n", result);
    printf("Execution Time: %.6f seconds\n", elapsed.count());
    return 0;
}
