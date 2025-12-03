#ifndef HASHTABLE_H
#define HASHTABLE_H

#include <vector>
#include <list>
#include <utility>
#include <string>
#include <algorithm> // For std::find_if

// max_prime is used by the helpful functions provided to you.
static const unsigned int max_prime = 1301081;
// the default_capacity is used if the initial capacity of the underlying vector of the hash table is zero.
static const unsigned int default_capacity = 11;

namespace cop4530 {

template <typename K, typename V>
class HashTable {
friend class PassServer;
public:

    size_t size() const;

    // Constructor: Initialize the hash table with a prime size <= specified size
    HashTable(size_t size = 101);
    
    // Destructor: Cleans up all dynamic allocations
    ~HashTable();

    // Check if key k is in the hash table
    bool contains(const K& k) const;

    // Check if key-value pair is in the hash table
    bool match(const std::pair<K, V>& kv) const;

    // Insert a key-value pair into the hash table
    bool insert(const std::pair<K, V>& kv);

    // Move version of insert
    bool insert(std::pair<K, V>&& kv);

    // Delete the key k and its corresponding value
    bool remove(const K& k);

    // Clear the hash table
    void clear();

    // Load the contents of a file into the hash table
    bool load(const char* filename);

    // Display all entries in the hash table
    void dump() const;

    // Write all elements in the hash table into a file
    bool write_to_file(const char* filename) const;

private:
    // Helper functions
    void makeEmpty();
    void rehash();
    size_t myhash(const K& k) const;
    unsigned long prime_below(unsigned long n) const;
    bool is_prime(unsigned long n) const;
    void setPrimes(std::vector<unsigned long>& vprimes) const;

    // Member variables
    std::vector<std::list<std::pair<K, V>>> theLists; // The array of Lists
    size_t currentSize; // Current number of elements
};

} // namespace cop4530

#include "hashtable.hpp" // Include the implementation of the template class

#endif // HASHTABLE_H

//#include "passserver.h"