#ifndef PASSSERVER_H
#define PASSSERVER_H

#include "hashtable.h"

#include <string>
#include <utility> // For std::pair

namespace cop4530 {

class PassServer {
public:
    // Constructor
    PassServer(size_t size = 101);

    // Destructor
    ~PassServer();

    // Load a password file into the HashTable object. Each line of the file will have a pair of username and password.
    bool load(const char* filename);

    bool addUser(std::pair<std::string, std::string>& kv);

    // Add a user with username and password. The password will be encrypted before storing.
    bool addUser(std::pair<std::string, std::string>&& kv);

    // Remove a user with the specified username
    bool removeUser(const std::string& k);

    // Change a user's password
    bool changePassword(const std::pair<std::string, std::string>& p, const std::string& newpassword);

    // Check if a user exists
    bool find(const std::string& user) const;

    // Show the structure and contents of the HashTable object
    void dump() const;

    // Return the size of the HashTable
    size_t size() const;

    // Save the username and password combinations into a file
    bool write_to_file(const char* filename) const;

private:
    // Encrypt the parameter str and return the encrypted string
    std::string encrypt(const std::string& str) const;

    // HashTable to store user and encrypted password pairs
    HashTable<std::string, std::string> table;
};

} // namespace cop4530

#endif // PASSSERVER_H