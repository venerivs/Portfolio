//------------------------------------------------------------------------------------------------
//Stack(): zero-argument constructor.
//------------------------------------------------------------------------------------------------

template<typename T>
TStack<T>::TStack() : topNode(nullptr), stackSize(0) {
    // creates an empty stack
}
//------------------------------------------------------------------------------------------------
//~Stack (): destructor.
//------------------------------------------------------------------------------------------------
template<typename T>
TStack<T>::~TStack() {
    Node<T>* currentNode = topNode;
    while(currentNode != nullptr) {
        Node<T>* nextNode = currentNode->prev; // moves down the stack once
        delete currentNode; // deletes the current node
        currentNode = nextNode; // updates node pointer
    }
    topNode = nullptr; // sets top node to null
    stackSize = 0; // resets size
}
//------------------------------------------------------------------------------------------------
//bool empty() const: returns true if the Stack contains no elements, and false otherwise.
//------------------------------------------------------------------------------------------------
template<typename T>
bool TStack<T>::empty() const {
    return topNode == nullptr;
}


//------------------------------------------------------------------------------------------------
//void clear(): delete all elements from the stack.
//Since the clear funciton only has one while loop, the worst case run scenario would be O(n)
//------------------------------------------------------------------------------------------------
template<typename T>
void TStack<T>::clear() {
    Node<T>* currentNode = topNode;
    while(currentNode != nullptr) {
        Node<T>* toDelete = currentNode;
        currentNode = currentNode->prev; // move to the next stack item
        delete toDelete; 
    }
    topNode = nullptr; // Reset the top pointer
    stackSize = 0; // set size to zero
}
//------------------------------------------------------------------------------------------------
//void push(const T& x): adds x to the Stack. copy version.
//------------------------------------------------------------------------------------------------
template<typename T>
void TStack<T>::push(const T& x) {
    Node<T>* newNode = new Node<T>(x, topNode); // create a new node with x data
    topNode = newNode; // turn topnode into new node
    stackSize++; 
}
//------------------------------------------------------------------------------------------------
//void push(T && x): adds x to the Stack. move version.
//------------------------------------------------------------------------------------------------
template<typename T>
void TStack<T>::push(T&& x) {
    Node<T>* newNode = new Node<T>(std::move(x), topNode); // Use std::move to convert x into an r-value
    topNode = newNode; // set new node to be top of stack
    stackSize++; 
}
//------------------------------------------------------------------------------------------------
//T& top(): returns a reference to the most recently added element of the Stack (as a modifiable L-value).
//------------------------------------------------------------------------------------------------
template<typename T>
T& TStack<T>::top() {
    if (empty()) {
        throw std::out_of_range("Stack is empty"); 
    }
    return topNode->data; // return topnodes data
}
//------------------------------------------------------------------------------------------------
//constT& top() const: accessor that returns the most recently added element of the Stack (as a const reference)
//------------------------------------------------------------------------------------------------
template<typename T>
const T& TStack<T>::top() const {
    if (empty()) {
        throw std::out_of_range("Stack is empty"); // or handle the error as appropriate for your application
    }
    return topNode->data; // Return the data stored in the top node
}
//------------------------------------------------------------------------------------------------
//void pop(): removes and discards the most recently added element of the Stack.
//------------------------------------------------------------------------------------------------
template<typename T>
void TStack<T>::pop() {
    if (empty()) {
        throw std::out_of_range("Cannot pop from an empty stack");
    }
    Node<T>* nodeToDelete = topNode;
    topNode = topNode->prev; // move pointer down 1
    delete nodeToDelete; // delete old node
    stackSize--; //decrease stack size
}
//------------------------------------------------------------------------------------------------
//int size() const: returns the number of elements stored in the Stack.
//------------------------------------------------------------------------------------------------
template<typename T>
int TStack<T>::size() const {
    return stackSize;
}
//------------------------------------------------------------------------------------------------
// void print(std::ostream& os, char ofc = ' ') const: print elements of Stack to ostream os. ofc
// is the separator between elements in the stack when they are printed out. Note that print() 
// prints elements in the opposite order of the Stack
//------------------------------------------------------------------------------------------------
template<typename T>
void TStack<T>::print(std::ostream& os, char ofc) const {
    for (Node<T>* currentNode = topNode; currentNode != nullptr; currentNode = currentNode->prev) {
        os << currentNode->data << ofc;
    }
    os << std::endl; // Optionally end with a newline
}