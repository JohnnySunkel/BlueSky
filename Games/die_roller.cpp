// Die Roller
// Demonstrates generating random numbers

#include "pch.h"
#include <iostream>
#include <cstdlib>
#include <ctime>

using namespace std;

int main()
{
    // seed the random number generator
	srand(static_cast<unsigned int>(time(0)));

	// generate a random number
	int randomNumber = rand();

	// get a number between 1 and 6
	int die = (randomNumber % 6) + 1;
	cout << "You rolled a " << die << endl;

	return 0;
}
