#include <iostream>
#include <string>
#include <cctype>
#include <stack>
#include <vector>
#include <sstream>

#include "TStack.h"

using namespace std;

bool isOperator(char c) {
    return c == '+' || c == '-' || c == '*' || c == '/';
}

bool isParenthesis(char c) {
    return c == '(' || c == ')';
}



int precedence(char op) {
    if (op == '+' || op == '-') return 1;
    if (op == '*' || op == '/') return 2;
    return 0;
}
// turns input stings into
vector<string> tokenize(const string& input) {
    stringstream ss(input);
    vector<string> tokens;
    string token;
    while (ss >> token) {
        tokens.push_back(token);
    }
    return tokens;
}


int evaluatePostfix(const string& postfix) {
    TStack<int> stack;
    stringstream ss(postfix);
    string token;

    while (ss >> token) {
        if (isdigit(token[0])) { // token operand
            stack.push(stoi(token));
        } else { // token operator
            int op2 = stack.top(); stack.pop(); // pop the operands
            int op1 = stack.top(); stack.pop();

            switch (token[0]) { // doing the math
                case '+': stack.push(op1 + op2); break;
                case '-': stack.push(op1 - op2); break;
                case '*': stack.push(op1 * op2); break;
                case '/': stack.push(op1 / op2); break; // 
            }
        }
    }

    return stack.top(); // returns the result
}

int main() {

    
    
    cout << "Enter an infix expression enter: \n";
    string input;
    getline(cin, input); // reads input
    cout << input << "\n";
   
    vector<string> tokens = tokenize(input);
    stack<string> s;
    string postfix = "";
    bool lastWasOp = true; // checks if equation entered will work

     


    for (const string& token : tokens) {
        if (isdigit(token[0]) || isalpha(token[0]) || (token.size() > 1 && token[1] == '.')) { // Operand
            postfix += token + " ";
            lastWasOp = false;
        } else if (token == "(") {
            s.push(token);
            lastWasOp = true;
        } else if (isOperator(token[0])) {
            if (lastWasOp) {
                cout << "Error: Operator without preceding operand." << endl;
                return -1;
            }
            while (!s.empty() && precedence(s.top()[0]) >= precedence(token[0])) {
                postfix += s.top() + " ";
                s.pop();
            }
            s.push(token);
            lastWasOp = true;
        } else if (token == ")") {
            bool foundOpenParenthesis = false;
            while (!s.empty() && s.top() != "(") {
                postfix += s.top() + " ";
                s.pop();
            }
            if (!s.empty() && s.top() == "(") {
                s.pop();
                foundOpenParenthesis = true;
            }
            if (!foundOpenParenthesis) {
                cout << "Error: Mismatched parentheses." << endl;
                return -1;
            }
            lastWasOp = false;
        } else {
            cout << "Error - Too few operands'" << token << "'." << endl;
            return -1;
        }
    }

    if (lastWasOp) {
        cout << "Error: Expression ends with an operator." << endl;
        return -1;
    }

    while (!s.empty()) {
        if (s.top() == "(") {
            cout << "Error: Mismatched parentheses." << endl;
            return -1;
        }
        postfix += s.top() + " ";
        s.pop();
    }

    cout << "Postfix expression: " << postfix << "\n" << endl;
    int result = evaluatePostfix(postfix);
    cout << "Result: " << result << endl;



    return 0;
}