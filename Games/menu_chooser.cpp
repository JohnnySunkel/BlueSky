// Menu Chooser
// Demonstrates the switch statement

#include "pch.h"
#include <iostream>

using namespace std;

int main()
{
    cout << "Difficulty levels\n\n";
	cout << "1 - Easy\n";
	cout << "2 - Normal\n";
	cout << "3 - Hard\n";

	int choice;
	cout << "Choice: ";
	cin >> choice;

	switch (choice)
	{
		case 1:
			cout << "You picked Easy.\n";
			break;
		case 2:
			cout << "You picked Normal\n.";
			break;
		case 3:
			cout << "You picked hard.\n";
			break;
		default:
			cout << "You made an illegal choice.\n";
	}
	return 0;
}
