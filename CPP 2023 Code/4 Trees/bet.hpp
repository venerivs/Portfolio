//--------------------------------------------------------------------------
//BET()
//--------------------------------------------------------------------------
template <typename T>
BET<T>::BET() : root{nullptr} {}
//--------------------------------------------------------------------------
//BET(const string& postfix)
//--------------------------------------------------------------------------
template <typename T>
BET<T>::BET(const string& postfix) 
{
    root = nullptr;
    buildFromPostfix(postfix);
}
//--------------------------------------------------------------------------
//BET(const BET& other)
//--------------------------------------------------------------------------
template <typename T>
BET<T>::BET(const BET& other) 
{
    root = clone(other.root);
}
//--------------------------------------------------------------------------
//~BET() -> deconstructor
//--------------------------------------------------------------------------

template <typename T>
BET<T>::~BET() 
{
    makeEmpty(root);
}
//--------------------------------------------------------------------------
//buildFromPostfix(const string& postfix)
//--------------------------------------------------------------------------


template <typename T>
bool BET<T>::buildFromPostfix(const string& postfix) {
    if (root != nullptr) makeEmpty(root); // Clear existing tree

    std::stack<BinaryNode*> stack;
    stringstream ss(postfix);
    string token;
    int operands = 0, operators = 0;

    while (ss >> token) {
        // Check if the token is an operand (simple check, assuming operands are digits or variables)
        if (isdigit(token[0]) || isalpha(token[0])) {
            stack.push(new BinaryNode(token));
            operands++;
        } else { // Token is assumed to be an operator
            operators++;
            if (stack.size() < 2) {
                cout << "Error: Operator \"" << token << "\" does not have enough operands." << endl;
                return false;
            }
            auto right = stack.top(); stack.pop();
            auto left = stack.top(); stack.pop();
            auto newNode = new BinaryNode(token, left, right);
            stack.push(newNode);
        }
    }

    // After processing all tokens, check for a valid expression
    // Valid if there's exactly one item in the stack and the number of operators and operands are appropriate
    if (stack.size() != 1 || operands - operators != 1) {
        cout << "Error: Operator does not have the corresponding operands." << endl;
        return false;
    }

    root = stack.top();
    return true; // Expression is valid
}


//--------------------------------------------------------------------------
//operator=(const BET& rhs)
//--------------------------------------------------------------------------

template <typename T>
const BET<T>& BET<T>::operator=(const BET& rhs) 
{
    if (this != &rhs) {
        makeEmpty(root);
        root = clone(rhs.root);
    }
    return *this;
}
//--------------------------------------------------------------------------
//printInfixExpression()
//--------------------------------------------------------------------------
template <typename T>
void BET<T>::printInfixExpression() 
{
    printInfixExpression(root);
    cout << endl; 
}
//--------------------------------------------------------------------------
//printPostfixExpression()
//--------------------------------------------------------------------------


template <typename T>
void BET<T>::printPostfixExpression() 
{
    printPostfixExpression(root);
    cout << endl; 
}
//--------------------------------------------------------------------------
//size()
//--------------------------------------------------------------------------

template <typename T>
size_t BET<T>::size() 
{
    return size(root);
}
//--------------------------------------------------------------------------
//leaf_nodes()
//--------------------------------------------------------------------------

template <typename T>
size_t BET<T>::leaf_nodes() 
{
    return leaf_nodes(root);
}
//--------------------------------------------------------------------------
//empty()
//--------------------------------------------------------------------------

template <typename T>
bool BET<T>::empty() 
{
    return root == nullptr;
}
//--------------------------------------------------------------------------
//printInfixExpression(BinaryNode* n)
//--------------------------------------------------------------------------

// Private methods implementation
template<typename T>
void BET<T>::printInfixExpression(BinaryNode* n) 
{
    if (n != nullptr) {
        if (n->left) cout << "( ";

        printInfixExpression(n->left);
        cout << n->element << " ";
        printInfixExpression(n->right);

        if (n->right) cout << ") ";
    }
}
//--------------------------------------------------------------------------
//makeEmpty(BinaryNode*& t)
//--------------------------------------------------------------------------

template<typename T>
void BET<T>::makeEmpty(BinaryNode*& t) 
{
    if (t != nullptr) {
        makeEmpty(t->left);
        makeEmpty(t->right);

        delete t;
        t = nullptr;
    }
}
//--------------------------------------------------------------------------
//clone(BinaryNode* t)
//--------------------------------------------------------------------------

template<typename T>
typename BET<T>::BinaryNode* BET<T>::clone(BinaryNode* t) 
{
    if (t == nullptr)
        return nullptr;
    else
        return new BinaryNode{t->element, clone(t->left), clone(t->right)};
}
//--------------------------------------------------------------------------
//printPostfixExpression(BinaryNode* n)
//--------------------------------------------------------------------------

template<typename T>
void BET<T>::printPostfixExpression(BinaryNode* n) 
{
    if (n != nullptr) {
        printPostfixExpression(n->left);
        printPostfixExpression(n->right);

        cout << n->element << " ";
    }
}
//--------------------------------------------------------------------------
//size(BinaryNode* t)
//--------------------------------------------------------------------------

template<typename T>
size_t BET<T>::size(BinaryNode* t) 
{
    if (t == nullptr)
        return 0;
    else
        return 1 + size(t->left) + size(t->right);
}
//--------------------------------------------------------------------------
//leaf_nodes(BinaryNode* t)
//--------------------------------------------------------------------------

template<typename T>
size_t BET<T>::leaf_nodes(BinaryNode* t) 
{
    if (t == nullptr)
        return 0;

    if (t->left == nullptr && t->right == nullptr)
        return 1;
        
    else
        return leaf_nodes(t->left) + leaf_nodes(t->right);
}
