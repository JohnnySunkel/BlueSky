#include <iostream>
#include <vector>
#include <algorithm>

using namespace std;

int main()
{
  cout << "Creating a 10 element vector to hold scores.\n";
	vector<int> scores(10, 0);  // initializes all 10 elements to 0
	cout << "Vector size is: " << scores.size() << endl;
	cout << "Vector capacity is: " << scores.capacity() << endl;

	cout << "Adding a score.\n";
	scores.push_back(0);  // memory is reallocated to accomodate growth
	cout << "Vector size is: " << scores.size() << endl;
	cout << "Vector capacity is: " << scores.capacity() << endl;
}
