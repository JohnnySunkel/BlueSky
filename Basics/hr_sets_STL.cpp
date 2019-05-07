#include <cmath>
#include <cstdio>
#include <vector>
#include <iostream>
#include <set>
#include <algorithm>
using namespace std;


int main() {
    int n;
    set <int> ss;
    cin >> n;
    for (int i = 0; i < n; i++) {
        int t, q;
        cin >> t >> q;
        switch(t) {
            case 1:
                ss.insert(q);
                break;
            case 2:
                ss.erase(q);
                break;
            case 3:
                cout << (ss.find(q) == ss.end() ? "No" : "Yes") << endl;
                break;
        }
    }   
    return 0;
}
