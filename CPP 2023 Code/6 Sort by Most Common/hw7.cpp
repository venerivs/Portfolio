#include <iostream>
#include <fstream>
#include <string>
#include <unordered_map>
#include <map>
#include <vector>
#include <algorithm>
#include <cctype>
#include <ctime>

using namespace std;

// Helper structure to track occurrences and first appearance
struct Counter {
    int count = 0;
    int first_appearance = 0;
};

int main(int argc, char* argv[]) {
    if (argc != 2) {
        cerr << "Usage: " << argv[0] << " <filename>" << endl;
        return 1;
    }

    ifstream file(argv[1]);
    if (!file) {
        cerr << "File cannot be opened: " << argv[1] << endl;
        return 1;
    }

    unordered_map<string, Counter> word_count;
    unordered_map<string, Counter> number_count;
    unordered_map<char, Counter> char_count;
    int order = 0; // Order of first appearance

    string token;
    char ch;
    while (file.get(ch)) {
        if (isalnum(ch)) {
            token += tolower(ch);
        } else {
            if (!token.empty()) {
                if (all_of(token.begin(), token.end(), ::isdigit)) {
                    if (number_count[token].count == 0) number_count[token].first_appearance = order++;
                    number_count[token].count++;
                } else {
                    if (word_count[token].count == 0) word_count[token].first_appearance = order++;
                    word_count[token].count++;
                }
                token.clear();
            }
            if (char_count[ch].count == 0) char_count[ch].first_appearance = order++;
            char_count[ch].count++;
        }
    }

    // Include the last token
    if (!token.empty()) {
        if (all_of(token.begin(), token.end(), ::isdigit)) {
            if (number_count[token].count == 0) number_count[token].first_appearance = order++;
            number_count[token].count++;
        } else {
            if (word_count[token].count == 0) word_count[token].first_appearance = order++;
            word_count[token].count++;
        }
    }

    file.close();

    // Sorting functions for different containers
    auto sortByFrequencyAndOrder = [](const pair<string, Counter>& a, const pair<string, Counter>& b) {
        if (a.second.count != b.second.count)
            return a.second.count > b.second.count;
        return a.second.first_appearance < b.second.first_appearance;
    };

    auto sortByCharFrequencyAndOrder = [](const pair<char, Counter>& a, const pair<char, Counter>& b) {
        if (a.second.count != b.second.count)
            return a.second.count > b.second.count;
        return a.first < b.first;  // Direct comparison since ASCII values are directly comparable
    };

    // Convert to vectors and sort
    vector<pair<string, Counter>> words(word_count.begin(), word_count.end());
    vector<pair<string, Counter>> numbers(number_count.begin(), number_count.end());
    vector<pair<char, Counter>> chars(char_count.begin(), char_count.end());

    sort(words.begin(), words.end(), sortByFrequencyAndOrder);
    sort(numbers.begin(), numbers.end(), sortByFrequencyAndOrder);
    sort(chars.begin(), chars.end(), sortByCharFrequencyAndOrder);

    // Output top 10 for each category

    cout << "\n----------------------------------------\n";
    cout << "Top 10 Characters:" << endl;
    cout << "----------------------------------------\n";
    for (int i = 0; i < 10 && i < chars.size(); ++i) {
        cout << "Character " << i << ": " << (chars[i].first == '\n' ? "\\n" :
                              chars[i].first == '\t' ? "\\t" :
                              chars[i].first == ' '  ? "space" : string(1, chars[i].first))
             << ", Frequency: " << chars[i].second.count
             << ", Priority: " << chars[i].second.first_appearance << endl;
    }

    cout << "\n----------------------------------------\n";
    cout << "Top 10 Words:" << endl;
    cout << "----------------------------------------\n";
    for (int i = 0; i < 10 && i < words.size(); ++i) {
        cout << "Word " << i << ": " << words[i].first
             << ", Frequency: " << words[i].second.count
             << ", Priority: " << words[i].second.first_appearance << endl;
    }

    cout << "\n----------------------------------------\n";
    cout << "Top 10 Numbers:" << endl;
    cout << "----------------------------------------\n";
    for (int i = 0; i < 10 && i < numbers.size(); ++i) {
        cout << "Number " << i << ": " << numbers[i].first
             << ", Frequency: " << numbers[i].second.count
             << ", Priority: " << numbers[i].second.first_appearance << endl;
    }

    return 0;
}