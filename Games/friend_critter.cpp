// Friend Critter
// Demonstrates friend functions and operator overloading

#include "pch.h"
#include <iostream>
#include <string>

using namespace std;

class Critter
{
	// make following global functions friends of the Critter class
	friend void Peek(const Critter& aCritter);
	friend ostream& operator<<(ostream& os, const Critter& aCritter);
public:
	Critter(const string& name = "");

private:
	string m_Name;
};

Critter::Critter(const string& name):
	m_Name(name)
{}

void Peek(const Critter& aCritter);
ostream& operator<<(ostream& os, const Critter& aCritter);

int main()
{
    Critter crit("Poochie");

	cout << "Calling Peek() to access crit's private data member, m_Name: \n";
	Peek(crit);

	cout << "\nSending crit object to cout with the << operator: \n";
	cout << crit;

	return 0;
}
