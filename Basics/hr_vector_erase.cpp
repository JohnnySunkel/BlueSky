#include <cmath>
#include <cstdio>
#include <vector>
#include <iostream>
#include <algorithm>
using namespace std;


int main() {
    int n;
    cin >> n;
    vector <int> v;
    int a;
    for (int i = 0; i < n; i++) {
        cin >> a;
        v.push_back(a);
    }
    int k;
    cin >> k;
    v.erase(v.begin() + k - 1);
    int l, m;
    cin >> l >> m;
    v.erase(v.begin() + l - 1, v.begin() + m - 1);
    cout << v.size() << endl;
    for (int i = 0; i < v.size(); i++) {
        cout << v[i] << " ";
    }   
    return 0;
}
