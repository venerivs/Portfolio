#ifndef TSTACK_H
#define TSTACK_H

#include <iostream>
#include "node.h" 

template <typename T>
class TStack
{
public:
   TStack(); // zero-argument constructor.
   ~TStack(); // destructor.
   bool empty() const; // returns true if the Stack contains no elements, and false otherwise.
   void clear(); // delete all elements from the stack.
   void push(const T& x); // adds x to the Stack. copy version.
   void push(T&& x); // adds x to the Stack. move version.
   void pop(); // removes and discards the most recently added element of the Stack.
   T& top(); //returns a reference to the most recently added element of the Stack (as a modifiable L-value).
   const T& top() const; //const: accessor that returns the most recently added element of the Stack (as a const reference)
   
   int size() const; // const: returns the number of elements stored in the Stack.
   void print(std::ostream& os, char ofc = ' ') const; // print elements of Stack to ostream os. ofc is the separator between elements in the stack when they are printed out

private:
   Node<T>* topNode; //points to top of stack
   int stackSize; // size of stack
};



#include "TStack.hpp"

#endif