
#ifndef HASHTABLE_HPP
#define HASHTABLE_HPP

#include "hashtable.h"
#include <fstream>
#include <iostream>

namespace cop4530 {

template <typename K, typename V>
size_t HashTable<K, V>::size() const 
{
        return currentSize;
}

// Constructor
template <typename K, typename V>
HashTable<K, V>::HashTable(size_t size) : currentSize{0} 
{
    theLists.resize(prime_below(size));
}

// Destructor
template <typename K, typename V>
HashTable<K, V>::~HashTable() 
{
    clear();
}


// Check if key k is in the hash table
template <typename K, typename V>
bool HashTable<K, V>::contains(const K& k) const 
{
    auto & whichList = theLists[myhash(k)];
    return std::find_if(whichList.begin(), whichList.end(),
                        [&k](const std::pair<K, V>& kv){ return kv.first == k; }) != whichList.end();
}

// Checks if the hash table contains an exact key-value pair match
template <typename K, typename V>
bool HashTable<K, V>::match(const std::pair<K, V>& kv) const 
{
    const auto& whichList = theLists[myhash(kv.first)]; // Find the correct bucket
    return std::find(whichList.begin(), whichList.end(), kv) != whichList.end(); // Check if kv is in the bucket
}

// Inserts a new key-value pair into the hash table
template <typename K, typename V>
bool HashTable<K, V>::insert(const std::pair<K, V>& kv) 
{
    auto & whichList = theLists[myhash(kv.first)];
    if (std::find_if(whichList.begin(), whichList.end(),
                     [&kv](const std::pair<K, V>& p){ return p.first == kv.first; }) != whichList.end())
        return false;
    whichList.push_back(kv);

    if (++currentSize > theLists.size())
        rehash();

    return true;
}

template <typename K, typename V>
bool HashTable<K, V>::insert(std::pair<K, V>&& kv) 
{
    auto & whichList = theLists[myhash(kv.first)];
    if (std::find_if(whichList.begin(), whichList.end(),
                     [&kv](const std::pair<K, V>& p){ return p.first == kv.first; }) != whichList.end())
        return false;
    whichList.push_back(std::move(kv));

    if (++currentSize > theLists.size())
        rehash();

    return true;
}

template <typename K, typename V>
bool HashTable<K, V>::remove(const K& k) 
{
    auto &whichList = theLists[myhash(k)]; // Find the correct bucket
    auto itr = std::find_if(whichList.begin(), whichList.end(), 
                            [&](const std::pair<K, V>& kv) { return kv.first == k; });
    if (itr == whichList.end()) {
        return false; // Key not found
    }
    whichList.erase(itr); // Remove the element
    --currentSize;
    return true;
}

template <typename K, typename V>
void HashTable<K, V>::clear() 
{
    for (auto &list : theLists) {
        list.clear(); // Clear each bucket
    }
    currentSize = 0; // Reset size
}

template <typename K, typename V>
bool HashTable<K, V>::load(const char* filename) 
{
    std::ifstream file(filename);
    if (!file) {
        return false; // Unable to open file
    }

    clear(); // Clear current hash table contents before loading new ones

    K key;
    V value;
    while (file >> key >> value) {
        insert(std::make_pair(std::move(key), std::move(value))); // Insert using move semantics
    }

    return true;
}

template <typename K, typename V>
void HashTable<K, V>::dump() const 
{
    for (size_t i = 0; i < theLists.size(); ++i) {
        std::cout << "Bucket " << i << ": ";
        auto &bucket = theLists[i];
        for (auto itr = bucket.begin(); itr != bucket.end(); ++itr) {
            if (itr != bucket.begin()) {
                std::cout << "; ";
            }
            std::cout << itr->first << " " << itr->second;
        }
        std::cout << std::endl;
    }
}

template <typename K, typename V>
bool HashTable<K, V>::write_to_file(const char* filename) const 
{
    std::ofstream file(filename);
    if (!file) {
        std::cerr << "Error opening file for writing: " << filename << std::endl;
        return false;
    }

    for (const auto &bucket : theLists) {
        for (const auto &kv : bucket) {
            // Assuming decryption is needed, replace `kv.second` with the decrypted value
            // For simplicity, this example writes the values as-is
            file << kv.first << " " << kv.second << std::endl;
        }
    }

    file.close();
    return true;
}


template <typename K, typename V>
void HashTable<K, V>::makeEmpty() 
{
    for (auto &thisList : theLists) {
        thisList.clear();
    }
    currentSize = 0; // Reset the size after clearing
}

template <typename K, typename V>
void HashTable<K, V>::rehash() 
{
    std::vector<std::list<std::pair<K, V>>> oldLists = theLists;
    // Resize theLists to the next prime number that is at least twice as large
    theLists.resize(prime_below(2 * oldLists.size()));
    for (auto &list : theLists) {
        list.clear(); // Clear new lists after resize
    }

    currentSize = 0; // Reset current size before re-inserting elements
    for (auto &list : oldLists) {
        for (auto &kv : list) {
            insert(std::move(kv)); // Re-insert elements into theLists
        }
    }
}

template <typename K, typename V>
size_t HashTable<K, V>::myhash(const K& k) const 
{
    static std::hash<K> hf;
    return hf(k) % theLists.size();
}

template <typename K, typename V>
unsigned long HashTable<K, V>::prime_below(unsigned long n) const 
{
    if (n > max_prime) {
        std::cerr << "Error: The number is too large to handle.\n";
        return 0;
    }
    if (n == 2) {
        return 2;
    }
    if (n % 2 == 0) {
        --n; // Ensure n is odd to simplify the search
    }
    for (unsigned long i = n; i >= 2; i -= 2) {
        if (is_prime(i)) return i;
    }
    return 2; // Fallback to the smallest prime if not found
}

template <typename K, typename V>
bool HashTable<K, V>::is_prime(unsigned long n) const
{
    if (n == 2 || n == 3) return true;
    if (n == 1 || n % 2 == 0) return false;
    for (unsigned long i = 3; i * i <= n; i += 2) {
        if (n % i == 0) return false;
    }
    return true;
}

template <typename K, typename V>
void HashTable<K, V>::setPrimes(std::vector<unsigned long>& vprimes) const 
{
    std::fill(vprimes.begin(), vprimes.end(), 1); // Assume all numbers are prime initially
    vprimes[0] = vprimes[1] = 0; // 0 and 1 are not prime
    for (unsigned long i = 2; i * i < vprimes.size(); ++i) {
        if (vprimes[i]) {
            for (unsigned long j = i * i; j < vprimes.size(); j += i) {
                vprimes[j] = 0; // Mark multiples of i as not prime
            }
        }
    }
}

}

#endif HASHTABLE_HPP