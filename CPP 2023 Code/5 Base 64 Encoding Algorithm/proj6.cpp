#include "passserver.cpp"
#include <iostream>
#include <fstream>
#include <utility>

using namespace std;
using namespace cop4530;

void Menu();

int main() 
{
    cout << "PassServer Demonstration" << endl;
    PassServer server(101); //server size
    
    char choice;
    string username, password, newpassword;
    pair<string, string> userPass;
    const char* filename = "userinfo.txt"; // file that saves usernames/passwords

    do {
        Menu();
        cin >> choice;
        switch(choice) {
            case 'a': // Add user
                cout << "Enter username: ";
                cin >> username;
                cout << "Enter password: ";
                cin >> password;
                userPass = make_pair(username, password);
                if(server.addUser(move(userPass))) {
                    cout << "User added successfully." << endl;
                } else {
                    cout << "User already exists or error adding user." << endl;
                }
                break;
            case 'r': // Remove user
                cout << "Enter username to remove: ";
                cin >> username;
                if(server.removeUser(username)) {
                    cout << "User removed successfully." << endl;
                } else {
                    cout << "Error removing user." << endl;
                }
                break;
            case 'c': // Change password
                cout << "Enter username: ";
                cin >> username;
                cout << "Enter old password: ";
                cin >> password;
                cout << "Enter new password: ";
                cin >> newpassword;
                if(server.changePassword(make_pair(username, password), newpassword)) {
                    cout << "Password changed successfully." << endl;
                } else {
                    cout << "Error changing password." << endl;
                }
                break;
            case 'f': // Find user
                cout << "Enter username to find: ";
                cin >> username;
                if(server.find(username)) {
                    cout << "User found." << endl;
                } else {
                    cout << "User not found." << endl;
                }
                break;
            case 'd': // Dump the hashtable
                server.dump();
                break;
            case 's': // Show size
                cout << "Size of hashtable: " << server.size() << endl;
                break;
            case 'w': // Write to file
                if(server.write_to_file(filename)) {
                    cout << "Data written to file successfully." << endl;
                } else {
                    cout << "Error writing to file." << endl;
                }
                break;
            case 'l': // Load from file
                if(server.load(filename)) {
                    cout << "Data loaded from file successfully." << endl;
                } else {
                    cout << "Error loading from file." << endl;
                }
                break;
            case 'x': // Quit
                cout << "Quitting program." << endl;
                return 0;
                break;
            default:
                cout << "Invalid option, please try again." << endl;
        }
    } while(choice != 'q');

    return 0;
}

void Menu() 
{
    cout << "\nMenu:" << endl;
    cout << "  l - Load From File" << endl;
    cout << "  a - Add User" << endl;
    cout << "  r - Remove User" << endl;
    cout << "  c - Change User Password" << endl;
    cout << "  f - Find User" << endl;
    cout << "  d - Dump HashTable" << endl;
    cout << "  s - HashTable Size" << endl;
    cout << "  w - Write to Password File -> userinfo.txt" << endl;
    cout << "  x - exit" << endl;
    cout << "Enter choice: ";
}
