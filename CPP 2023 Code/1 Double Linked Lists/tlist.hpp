template <typename T>   // Create empty linked list
TList<T>::TList()
{
        first = nullptr;
        last = nullptr;
        size = 0;
}

template <typename T>   // Create list with num copies of val
TList<T>::TList(T val, int num)
{
        first = last = nullptr;
        size = 0;


        for (int i = 0; i < num; ++i) {
                InsertBack(val);
        }
}

template <typename T>
TList<T>::~TList()
{
        static Node<T>* current = first;  //start at first node

        while (current != nullptr) {
                Node<T>* nextNode = current->next;  // Save the next node
                delete current;  // Delete the current node
                current = nextNode;  // Move to the next node
    }

    //delete all nodes, reset pointers and
        first = last = nullptr;
        size = 0;
}

template <typename T>
TList<T>& TList<T>:: operator=(const TList& L) //copy assignment operator
{
        if (this != &L) {
                // clear list
                Clear();

        // copy elements
                TListIterator<T> itr = L.GetIterator();
                while (itr.HasNext()) {
                InsertBack(itr.GetData());
                itr = itr.Next();
                }
    }
    return *this;
}

template <typename T>   //copy constructor
TList<T>::TList(const TList& L)
{
        *this = L;
}


template <typename T>   //move assignment operator
TList<T>& TList<T>:: operator=(TList && L)
{
        if (this != &L) {
                // clear list
                Clear();

                // move resources
                first = L.first;
                last = L.last;
                size = L.size;

                // set pointers to null
                L.first = L.last = nullptr;
                L.size = 0;
        }
    return *this;
}




template <typename T>   //checks if list is empty
bool TList<T>::IsEmpty() const
{
    char empt[5] = {'E','m','p','t','y'};
    char notempt[3] = {'N','o','t'};
        if (size == 0) 
        {
                return empt;
        }

        else
        {
                return notempt;   
        }
        
}





template <typename T>   //clears the list
void TList<T>::Clear       ()
{
        static Node<T>* current = first;
        while (current != nullptr) {
                Node<T>* nextNode = current->next;  
                delete current;  
                current = nextNode;  
        }
        first = last = nullptr;
        size = 0;
}

template <typename T>   //returns size of list
int TList<T>::GetSize() const
{
        return size;
}

template <typename T>   // Insert data at the front of the list
void TList<T>::InsertFront(const T& d)
{
static Node<T>* newnode;  
    newnode = new Node<T>(d);  

    if (IsEmpty()) {    //checks if list is empty

        first = last = newnode;
        size++;
    } else {          //inserts at front of list
        newnode->prev = nullptr; 
        newnode->next = first;
        first->prev = newnode;
        first = newnode;
        size++;
    }

}       


template <typename T>   // Insert data at the back of the list
void TList<T>::InsertBack(const T& d)
{
        static Node<T> * newnode;
        newnode = new Node<T>(d);
        if (size == 0) {    //checks if list is 0
                first = last = newnode;
                size++;
                return;
        }
        else {      //inserts at back of list
                last->next = newnode;
                newnode->prev = last;
                last = newnode;
                size++;
                return;
        }
}

template <typename T>   // Remove the front element of the list 
void TList<T>::RemoveFront()    
{
        if (size > 0) {
                //saves first node
                Node<T>* temp = first;

                //moves pointer up one
                first = first->next;

                // if list clears, pointer points at null
                if (size == 1) {
                        last = nullptr;
                }

        // old node deleted
                delete temp;

                --size;
        }
}

template <typename T>   // Remove the back element of the list
void TList<T>::RemoveBack()
{
if (size > 0) { 
        // saves last node pointer
        Node<T>* temp = last;

        if (size > 1) {
            // update pointer to previous node
            last = last->prev;
            last->next = nullptr;
        } else {
            first = last = nullptr;
        }

        delete temp;
        --size;
    }
}

template <typename T>
T& TList<T>::GetFirst() const // gets the first node
{

    if (first != nullptr) {
        return first->data;
    } else {
        return dummy;
    }
}

template <typename T>
T& TList<T>::GetLast() const    //returns the last node
{
    // Check if the list is not empty
    if (last != nullptr) {

        return last->data;
    } else {

        return dummy;
    }
}

template <typename T>
TListIterator<T> TList<T>::GetIterator() const
{
        TListIterator<T> itr;
        itr.ptr = first;
        return itr;
}

template <typename T>
TListIterator<T> TList<T>::GetIteratorEnd() const
{
    TListIterator<T> endItr;
    endItr.ptr = nullptr;
    return endItr;
}

template <typename T>
void TList<T>::Insert(TListIterator<T> pos, const T& d)
{
    // create a new node
    Node<T>* newNode = new Node<T>(d);

    // Checks if list is empty
    if (size == 0) {
        // set node
        first = last = newNode;
    } else {
        // update pointers
        if (pos.ptr == nullptr) {
            // insert at end
            last->next = newNode;
            newNode->prev = last;
            last = newNode;
        } else if (pos.ptr == first) {
            // insert at front
            newNode->next = first;
            first->prev = newNode;
            first = newNode;
        } else {
            // insert somewhere inbetween
            newNode->prev = pos.ptr->prev;
            newNode->next = pos.ptr;
            pos.ptr->prev->next = newNode;
            pos.ptr->prev = newNode;
        }
    }

    ++size;
}




template <typename T>
TListIterator<T> TList<T>::Remove(TListIterator<T> pos)
{
    if (size > 0 && pos.ptr != nullptr) {
        Node<T>* removedNode = pos.ptr;

        // get next node
        TListIterator<T> nextItr;
        nextItr.ptr = removedNode->next;

        // update pointers
        if (removedNode->prev != nullptr) {
            removedNode->prev->next = removedNode->next;
        } else {
            // update first node
            first = removedNode->next;
        }

        if (removedNode->next != nullptr) {
            removedNode->next->prev = removedNode->prev;
        } else {
            // update last mode if removed.
            last = removedNode->prev;
        }


        delete removedNode;


        --size;


        return nextItr;
    }

    return GetIteratorEnd();
}





template <typename T>
void TList<T>::Print(std::ostream& os, char delim) const
{
    Node<T>* current = first;  

    while (current != nullptr) {
        os << current->data;  // print node data

        if (current->next != nullptr) {
            os << delim;  // print the delim
        }

        current = current->next;  //move to the next node
    }

    os << std::endl;  // print the end
}

template <typename T>
TListIterator<T>::TListIterator()
{
    // Initializes pointer to nullpointer
    ptr = nullptr;
}

template <typename T>
bool TListIterator<T>::HasNext() const
{
    // check if at end of list
    return (ptr != nullptr);
}

template <typename T>
bool TListIterator<T>::HasPrevious() const
{
    // check if at beginning of list
    return (ptr != nullptr && ptr->prev != nullptr);
}

template <typename T>
TListIterator<T> TListIterator<T>::Next()
{
    TListIterator<T> nextIter;
    
    // checks for a next item in list
    if (HasNext()) {
        nextIter.ptr = ptr->next;
    } else {
        // if no new item, set as last.
        nextIter.ptr = nullptr;
    }

    return nextIter;
}

template <typename T>
TListIterator<T> TListIterator<T>::Previous()
{
    TListIterator<T> prevIter;

    // check if previous node
    if (HasPrevious()) {
        prevIter.ptr = ptr->prev;
    } else {
        // if no previous node, set null
        prevIter.ptr = nullptr;
    }

    return prevIter;
}

template <typename T>
T& TListIterator<T>::GetData() const
{
    // Gets data
    return ptr->data;
}

template <typename T>
TList<T> operator+(const TList<T>& t1, const TList<T>& t2)
{
    TList<T> result;

    // Insert elements from the first list
    TListIterator<T> itr1 = t1.GetIterator();
    while (itr1.HasNext()) {
        result.InsertBack(itr1.GetData());
        itr1 = itr1.Next();
    }

    // Insert elements from the second list
    TListIterator<T> itr2 = t2.GetIterator();
    while (itr2.HasNext()) {
        result.InsertBack(itr2.GetData());
        itr2 = itr2.Next();
    }

    return result;
}