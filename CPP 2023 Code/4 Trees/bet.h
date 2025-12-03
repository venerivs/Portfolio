

#include <iostream>
#include <string>
#include <stack>
#include <sstream>
#include <vector>

using namespace std;

template <typename T>
class BET {
public:
    BET(); // default zero-parameter constructor. Builds an empty tree.
    BET(const string& postfix); // one-parameter constructor, where
                                  //parameter "postfix" is string containing a postfix expression. The tree
                                  //should be built based on the postfix expression. Tokens in the postfix
                                  //expression are separated by spaces.
    BET(const BET&); // copy constructor -- makes appropriate deep copy of
                        //the tree
    ~BET(); // destructor -- cleans up all dynamic space in the tree

    bool buildFromPostfix(const string& postfix); //parameter "postfix" is string containing a postfix expression. 
    const BET& operator=(const BET&); // assignment operator -- makes appropriate deep copy
    void printInfixExpression(); // Print the infix expression
    void printPostfixExpression(); // Print the postfix expression
    size_t size(); // Return the number of nodes in the tree
    size_t leaf_nodes(); // Return the number of leaf nodes in the tree
    bool empty(); // Return true if the tree is empty

private:
    struct BinaryNode {
        T element;
        BinaryNode *left;
        BinaryNode *right;

        BinaryNode(const T& theElement, BinaryNode* lt = nullptr, BinaryNode* rt = nullptr)
        : element{theElement}, left{lt}, right{rt} {}
    };

    BinaryNode* root; // Pointer to the root of the tree

    void printInfixExpression(BinaryNode* n); // print to the standard output the corresponding infix expression.
    void makeEmpty(BinaryNode*& t); // Delete all nodes in the subtree pointed to by t
    BinaryNode* clone(BinaryNode* t); // Clone all nodes in the subtree pointed to by t
    void printPostfixExpression(BinaryNode* n); // Private version to print postfix expression
    size_t size(BinaryNode* t); // Return the number of nodes in the subtree pointed to by t
    size_t leaf_nodes(BinaryNode* t); // Return the number of leaf nodes in the subtree pointed to by t
};


#include "bet.hpp"