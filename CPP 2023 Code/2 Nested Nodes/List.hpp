template<typename T>
List<T>::List() {
    init();
}

//**********************************
//**********************************
// Default constructor for const_iterator
template<typename T>
List<T>::const_iterator::const_iterator() : current(nullptr) {
    // Intentionally left blank
}

//**********************************
//**********************************

// Dereference operator for const_iterator
template<typename T>
const T& List<T>::const_iterator::operator*() const {
    if (current == nullptr) {
        throw std::out_of_range("Attempt to dereference null iterator.");
    }
    return current->data; // Directly access the data of the node
}

//**********************************
//**********************************
template<typename T>
typename List<T>::const_iterator& List<T>::const_iterator::operator++() {
    if (current == nullptr) {
        throw std::out_of_range("Incrementing a null iterator.");
    }
    current = current->next;
    return *this;
}


//**********************************
//**********************************

template<typename T>
typename List<T>::const_iterator List<T>::const_iterator::operator++(int) {
    const_iterator temp = *this;
    ++(*this); // Use the pre-increment
    return temp;
}

//**********************************
//**********************************
template<typename T>
typename List<T>::const_iterator& List<T>::const_iterator::operator--() {
    if (current == nullptr || current->prev == nullptr) {
        throw std::out_of_range("Decrementing a null or beginning iterator.");
    }
    current = current->prev;
    return *this;
}

//**********************************
//**********************************
template<typename T>
typename List<T>::const_iterator List<T>::const_iterator::operator--(int) {
    const_iterator temp = *this;
    --(*this); // Use the pre-decrement
    return temp;
}

//**********************************
//**********************************
template<typename T>
bool List<T>::const_iterator::operator==(const const_iterator &rhs) const {
    return current == rhs.current;
}

//**********************************
//**********************************
template<typename T>
bool List<T>::const_iterator::operator!=(const const_iterator &rhs) const {
    return current != rhs.current;
}

//**********************************
//**********************************
template<typename T>
T& List<T>::const_iterator::retrieve() const {
    if (current == nullptr) {
        throw std::out_of_range("Attempt to retrieve from an invalid iterator.");
    }
    return current->data;
}

//**********************************
//**********************************
template<typename T>
List<T>::const_iterator::const_iterator(Node *p) : current(p) {
    // Body intentionally left blank
}

//**********************************
//**********************************
template<typename T>
List<T>::iterator::iterator() : const_iterator() { }

//**********************************
//**********************************
template<typename T>
T& List<T>::iterator::operator*() {
    return const_iterator::retrieve(); // Use retrieve() to access the element, assuming retrieve() is accessible and returns T&
}

//**********************************
//**********************************
template<typename T>
const T& List<T>::iterator::operator*() const {
    // Explicitly call the const version of retrieve() from const_iterator
    return const_iterator::retrieve();
}

//**********************************
//**********************************
template<typename T>
typename List<T>::iterator& List<T>::iterator::operator++() {
    // Ensure we're not trying to increment a null iterator or one that's already at the end
    if (this->current == nullptr) {
        throw std::out_of_range("Incrementing a null iterator.");
    }
    this->current = this->current->next;
    return *this;
}

//**********************************
//**********************************
template<typename T>
typename List<T>::iterator List<T>::iterator::operator++(int) {
    iterator temp = *this;
    ++(*this); // Increment the iterator using the pre-increment operator
    return temp;
}

//**********************************
//**********************************
template<typename T>
typename List<T>::iterator& List<T>::iterator::operator--() {
    // Ensure we're not trying to decrement a null iterator or one that's before the beginning
    if (this->current == nullptr || this->current->prev == nullptr) {
        throw std::out_of_range("Decrementing a null or invalid iterator.");
    }
    this->current = this->current->prev;
    return *this;
}

//**********************************
//**********************************
template<typename T>
typename List<T>::iterator List<T>::iterator::operator--(int) {
    iterator temp = *this;
    --(*this); // Decrement the iterator using the pre-decrement operator
    return temp;
}

//**********************************
//**********************************
template<typename T>
List<T>::iterator::iterator(Node *p) : const_iterator(p) {
    // The constructor body is empty because all initialization is done by
    // the const_iterator's constructor, which initializes the 'current' pointer.
}

//**********************************
//**********************************
template<typename T>
List<T>::List(const List &rhs) : theSize(0), head(nullptr), tail(nullptr) {
    init(); // Initialize the list with the sentinel nodes
    for (const_iterator itr = rhs.begin(); itr != rhs.end(); ++itr) {
        push_back(*itr); // Copy elements from rhs to this list
    }
}

//**********************************
//**********************************
template<typename T>
List<T>::List(List && rhs) : theSize(rhs.theSize), head(rhs.head), tail(rhs.tail) {
    rhs.theSize = 0; // Reset the original list
    rhs.head = nullptr; // Prevent the original list from accessing the transferred nodes
    rhs.tail = nullptr;
}

//**********************************
//**********************************
template<typename T>
List<T>::List(int num, const T& val) : theSize(0), head(nullptr), tail(nullptr) {
    init(); // Initialize list with sentinel nodes
    for (int i = 0; i < num; ++i) {
        push_back(val); // Add 'num' elements of 'val' to the list
    }
}

//**********************************
//**********************************
template<typename T>
List<T>::List(const_iterator start, const_iterator end) : theSize(0), head(nullptr), tail(nullptr) {
    init(); // Initialize list with sentinel nodes
    for (const_iterator itr = start; itr != end; ++itr) {
        push_back(*itr); // Copy elements from the range into the new list
    }
}

//**********************************
//**********************************
template<typename T>
List<T>::List(std::initializer_list<T> iList) : theSize(0), head(nullptr), tail(nullptr) {
    init(); // Initialize list with sentinel nodes
    for (const T& item : iList) {
        push_back(item); // Add elements from initializer_list to the list
    }
}

//**********************************
//**********************************
template<typename T>
List<T>::~List() {
    clear(); // Delete all elements
    delete head; // Delete head sentinel node
    delete tail; // Delete tail sentinel node
}
//**********************************
//**********************************
template<typename T>
const List<T>& List<T>::operator=(const List<T> &rhs) {
    if (this != &rhs) { // Protect against self-assignment
        clear(); // Clear current contents
        // Copy elements from rhs
        for (const_iterator itr = rhs.begin(); itr != rhs.end(); ++itr) {
            push_back(*itr);
        }
    }
    return *this; // Return a reference to the current object
}
//**********************************
//**********************************
template<typename T>
List<T>& List<T>::operator=(List<T> &&rhs) {
    if (this != &rhs) { // Protect against self-assignment
        clear(); // Clear current contents
        delete head; // Delete existing sentinel nodes
        delete tail;

        // Transfer ownership of resources
        head = rhs.head;
        tail = rhs.tail;
        theSize = rhs.theSize;

        // Leave rhs in a valid state
        rhs.head = rhs.tail = nullptr;
        rhs.theSize = 0;
    }
    return *this; // Return a reference to the current object
}
//**********************************
//**********************************
template<typename T>
List<T>& List<T>::operator=(std::initializer_list<T> iList) {
    clear(); // Clear current contents
    // Add elements from initializer_list
    for (const T& item : iList) {
        push_back(item);
    }
    return *this; // Return a reference to the current object
}
//**********************************
//**********************************
template<typename T>
int List<T>::size() const {
    return theSize;
}
//**********************************
//**********************************
template<typename T>
bool List<T>::empty() const {
    return theSize == 0;
}
//**********************************
//**********************************
template<typename T>
void List<T>::clear() {
    while (!empty()) {
        pop_front(); // Repeatedly remove elements from the front
    }
}
//**********************************
//**********************************
template<typename T>
void List<T>::reverse() {
    if (empty() || theSize == 1) return; // No need to reverse if list is empty or has one element

    Node *current = head->next; // Start with the first actual element
    while (current != tail) {
        Node *next = current->next;
        // Swap the next and prev pointers of the current node
        current->next = current->prev;
        current->prev = next;
        current = next;
    }
    // Swap the head and tail sentinel nodes
    std::swap(head->next, tail->prev);
    if (head->next) head->next->prev = head; // Fix the prev pointer of the new first node
    if (tail->prev) tail->prev->next = tail; // Fix the next pointer of the new last node
}
//**********************************
//**********************************
template<typename T>
T& List<T>::front() {
    if (empty()) {
        throw std::out_of_range("Called front on empty list.");
    }
    return head->next->data; // head->next is the first actual element
}
//**********************************
//**********************************
template<typename T>
const T& List<T>::front() const {
    if (empty()) {
        throw std::out_of_range("Called front on empty list.");
    }
    return head->next->data; // head->next is the first actual element
}

//**********************************
//**********************************
template<typename T>
T& List<T>::back() {
    if (empty()) {
        throw std::out_of_range("Called back on empty list.");
    }
    return tail->prev->data; // tail->prev is the last actual element
}
//**********************************
//**********************************
template<typename T>
const T& List<T>::back() const {
    if (empty()) {
        throw std::out_of_range("Called back on empty list.");
    }
    return tail->prev->data; // tail->prev is the last actual element
}
//**********************************
//**********************************
template<typename T>
void List<T>::push_front(const T &val) {
    // Create a new node with val, where the new node's next is the current first node
    Node* newNode = new Node(val, head, head->next);
    head->next->prev = newNode; // Update the previous first node's prev to the new node
    head->next = newNode; // Update the head's next to the new node
    ++theSize; // Increment the size of the list
}
//**********************************
//**********************************
template<typename T>
void List<T>::push_front(T &&val) {
    // Similar to the copy version, but use std::move to construct the node
    Node* newNode = new Node(std::move(val), head, head->next);
    head->next->prev = newNode;
    head->next = newNode;
    ++theSize;
}
//**********************************
//**********************************
template<typename T>
void List<T>::push_back(const T &val) {
    // Create a new node with val, where the new node's previous is the current last node
    Node* newNode = new Node(val, tail->prev, tail);
    tail->prev->next = newNode; // Update the previous last node's next to the new node
    tail->prev = newNode; // Update the tail's prev to the new node
    ++theSize; // Increment the size of the list
}
//**********************************
//**********************************
template<typename T>
void List<T>::push_back(T &&val) {
    // Similar to the copy version, but use std::move to construct the node
    Node* newNode = new Node(std::move(val), tail->prev, tail);
    tail->prev->next = newNode;
    tail->prev = newNode;
    ++theSize;
}
//**********************************
//**********************************
template<typename T>
void List<T>::pop_front() {
    if (empty()) {
        throw std::out_of_range("Attempt to pop from an empty list.");
    }
    Node* oldNode = head->next; // The first actual element
    head->next = oldNode->next; // Update head's next to skip over the old first element
    oldNode->next->prev = head; // Update the new first element's prev to head
    delete oldNode; // Free the removed node
    --theSize; // Decrement the size of the list
}
//**********************************
//**********************************
template<typename T>
void List<T>::pop_back() {
    if (empty()) {
        throw std::out_of_range("Attempt to pop from an empty list.");
    }
    Node* oldNode = tail->prev; // The last actual element
    tail->prev = oldNode->prev; // Update tail's prev to skip over the old last element
    oldNode->prev->next = tail; // Update the new last element's next to tail
    delete oldNode; // Free the removed node
    --theSize; // Decrement the size of the list
}
//**********************************
//**********************************
template<typename T>
void List<T>::remove(const T &val) {
    Node* current = head->next; // Start with the first actual element
    while (current != tail) { // Iterate until the tail sentinel is reached
        if (current->data == val) {
            Node* toDelete = current;
            current->prev->next = current->next; // Bridge the previous node to the next
            current->next->prev = current->prev; // Bridge the next node to the previous
            current = current->next; // Move to the next node before deleting
            delete toDelete; // Delete the current node
            --theSize; // Decrement the size for each removed element
        } else {
            current = current->next; // Move to the next node if not deleting
        }
    }
}
//**********************************
//**********************************
template<typename T>
template<typename PREDICATE>
void List<T>::remove_if(PREDICATE pred) {
    Node* current = head->next; // Start with the first actual element
    while (current != tail) { // Iterate until the tail sentinel is reached
        if (pred(current->data)) {
            Node* toDelete = current;
            current->prev->next = current->next; // Bridge the previous node to the next
            current->next->prev = current->prev; // Bridge the next node to the previous
            current = current->next; // Move to the next node before deleting
            delete toDelete; // Delete the current node
            --theSize; // Decrement the size for each removed element
        } else {
            current = current->next; // Move to the next node if not deleting
        }
    }
}
//**********************************
//**********************************
template<typename T>
void List<T>::print(std::ostream& os, char ofc) const {
    for (const_iterator itr = begin(); itr != end(); ++itr) {
        os << *itr; // Output the current element
        auto nextItr = itr;
        ++nextItr;
        if (nextItr != end()) {
            os << ofc; // Output the delimiter if this is not the last element
        }
    }
}
//**********************************
//**********************************
template<typename T>
typename List<T>::iterator List<T>::begin() {
    return iterator(head->next); // head->next points to the first actual element
}
//**********************************
//**********************************
template<typename T>
typename List<T>::const_iterator List<T>::begin() const {
    return const_iterator(head->next); // head->next points to the first actual element
}

//**********************************
//**********************************
template<typename T>
typename List<T>::iterator List<T>::end() {
    return iterator(tail); // tail acts as the end marker
}

//**********************************
//**********************************
template<typename T>
typename List<T>::const_iterator List<T>::end() const {
    return const_iterator(tail); // tail is the end marker
}
//**********************************
//**********************************
template<typename T>
typename List<T>::iterator List<T>::insert(iterator itr, const T& val) {
    Node* p = itr.current; // The node at the iterator's current position
    ++theSize;
    Node* newNode = new Node(val, p->prev, p); // Create a new node with val
    p->prev->next = newNode; // Link the new node with the previous node
    p->prev = newNode; // Link the new node with p
    return iterator(newNode);
}
//**********************************
//**********************************
template<typename T>
typename List<T>::iterator List<T>::insert(iterator itr, T && val) {
    Node* p = itr.current;
    ++theSize;
    Node* newNode = new Node(std::move(val), p->prev, p); // Use std::move for val
    p->prev->next = newNode;
    p->prev = newNode;
    return iterator(newNode);
}
//**********************************
//**********************************
template<typename T>
typename List<T>::iterator List<T>::erase(iterator itr) {
    if (itr.current == nullptr || itr.current == tail) { // Guard against invalid iterator and tail
        throw std::out_of_range("Erase called with an invalid iterator.");
    }
    Node* p = itr.current;
    iterator retVal(p->next);
    p->prev->next = p->next;
    p->next->prev = p->prev;
    delete p; // Delete the node
    --theSize;
    return retVal; // Return iterator to the next element
}
//**********************************
//**********************************
template<typename T>
typename List<T>::iterator List<T>::erase(iterator start, iterator end) {
    while (start != end) {
        start = erase(start); // erase() updates start to the next element
    }
    return start; // or end, both are equivalent here
}
//**********************************
//**********************************
template<typename T>
bool operator==(const List<T>& lhs, const List<T>& rhs) {
    // First, check if the sizes of the two lists are the same
    if (lhs.size() != rhs.size()) {
        return false; // Lists can't be equal if their sizes differ
    }

    // Compare elements one by one using iterators
    typename List<T>::const_iterator itL = lhs.begin();
    typename List<T>::const_iterator itR = rhs.begin();
    while (itL != lhs.end() && itR != rhs.end()) {
        if (*itL != *itR) {
            return false; // Found elements that are not equal
        }
        ++itL;
        ++itR;
    }

    return true; // All elements are equal
}
//**********************************
//**********************************
template<typename T>
bool operator!=(const List<T>& lhs, const List<T>& rhs) {
    return !(lhs == rhs); // Return the negation of the equality comparison
}

//**********************************
//**********************************
template<typename T>
std::ostream& operator<<(std::ostream &os, const List<T> &l) {
    os << '['; // Start with an opening bracket to denote the start of the list
    typename List<T>::const_iterator it = l.begin();
    if (it != l.end()) { // Check if the list is not empty to handle the first element
        os << *it; // Insert the first element to avoid a trailing comma
        ++it;
    }
    while (it != l.end()) { // Iterate through the rest of the list
        os << ", " << *it; // Insert a comma before all elements after the first
        ++it;
    }
    os << ']'; // End with a closing bracket
    return os; // Return the ostream object to allow chaining of stream insertions
}

//**********************************
//**********************************

// Initialize the list
template<typename T>
void List<T>::init() {
    theSize = 0; // Set the initial size to 0
    head = new Node; // Create a new node for head
    tail = new Node; // Create a new node for tail

    // Now, link the head and tail sentinel nodes together
    head->next = tail; // head points to tail
    tail->prev = head; // tail points back to head
    head->prev = nullptr; // head's prev is null as it's the first node
    tail->next = nullptr; // tail's next is null as it's the last node
}