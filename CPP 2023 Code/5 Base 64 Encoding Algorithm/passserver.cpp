#include "passserver.h"
#include "base64.cpp"
#include <fstream>
#include <sstream> 

using namespace std;
using namespace cop4530;

// Constructor
PassServer::PassServer(size_t size) : table(size) {}

// Destructor
PassServer::~PassServer() {}

// Load username and password pairs from a file
bool PassServer::load(const char* filename) 
{
    std::ifstream file(filename);
    if (!file) return false;

    std::string username, password;
    while (file >> username >> password) {
        // Encrypt password before insertion
        addUser(std::make_pair(username, encrypt(password)));
    }
    return true;
}

// Add a new username and password
bool PassServer::addUser(std::pair<std::string, std::string>& kv) 
{
    kv.second = encrypt(kv.second); // Encrypt password before insertion
    return table.insert(kv);
}

// Add a new username and password
bool PassServer::addUser(std::pair<std::string, std::string>&& kv) 
{
    kv.second = encrypt(kv.second); // Encrypt password before insertion
    return table.insert(std::move(kv));
}

// Remove user
bool PassServer::removeUser(const std::string& k) 
{
    return table.remove(k);
}

// Change a user's password
bool PassServer::changePassword(const std::pair<std::string, std::string>& p, const std::string& newpassword) 
{
    if (!find(p.first)) return false; // User does not exist
    
    std::string encryptedOldPassword = encrypt(p.second);
    std::string encryptedNewPassword = encrypt(newpassword);

    if (encryptedOldPassword == encryptedNewPassword) return false; // New password must be different

    if (!table.match({p.first, encryptedOldPassword})) return false; // Old password does not match

    // Remove the old entry and add the new entry
    table.remove(p.first);
    return table.insert({p.first, encryptedNewPassword});
}

// Check if a user exists
bool PassServer::find(const std::string& user) const 
{
    return table.contains(user);
}

// Display all entries
void PassServer::dump() const 
{
    table.dump();
}

// Return the size
size_t PassServer::size() const 
{
    return table.size();
}


bool PassServer::write_to_file(const char* filename) const 
{
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Cannot open file for writing: " << filename << std::endl;
        return false;
    }

    // Assuming PassServer is a friend of HashTable or has some mechanism to access its elements
    for (const auto& list : table.theLists) { // Access each bucket of the hash table
        for (const auto& pair : list) { // Iterate over elements in the bucket
            file << pair.first << " " << pair.second << std::endl; // Write username and encrypted password
        }
    }

    file.close();
    return true;
}

// Encrypt password
std::string PassServer::encrypt(const std::string& str) const 
{
    const size_t MAX_ENCODED_LENGTH = (str.length() + 2 - ((str.length() + 2) % 3)) * 4 / 3; // Calculate max string length
    BYTE in[str.length() + 1];
    BYTE out[MAX_ENCODED_LENGTH + 1]; 
    
    std::strcpy(reinterpret_cast<char*>(in), str.c_str()); // Copy the std::string to BYTE array
    size_t outLength = base64_encode(in, out, std::strlen(reinterpret_cast<char*>(in)), 1); 
    out[outLength] = '\0'; // Ensure null termination

    return std::string(reinterpret_cast<char*>(out)); // Convert back to std::string
}

