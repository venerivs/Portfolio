#ifndef NODE_H
#define NODE_H

template <typename T>
class Node
{
public:
    Node(const T& data, Node<T>* prev = nullptr) : data(data), prev(prev) {}
    Node(T&& data, Node<T>* prev = nullptr) : data(std::move(data)), prev(prev) {}

    T data;        // data stored in node
    Node<T>* prev; // points to prev node

private:
    
};

#endif 