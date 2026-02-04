#ifndef csv_functions.h
#define csv_functions.h

#include <string>
#include <iostream>
#include <fstream>
#include <vector>

template<typename T>
void write_to_csv(T obj, std::string filename) {
    std::ofstream csv_file;    
    csv_file.open(filename);
    csv_file << obj << std::endl;
    csv_file.close();
}

template<typename T>
void write_to_csv(T obj, std::string filename, std::string* columns, std::size_t n) {
    std::ofstream csv_file;    
    csv_file.open(filename);
    for(std::size_t index = 0; index < n; index++) {
        csv_file << columns[index];
        if(index < n - 1) csv_file << ' ';
    }
    csv_file << '\n';
    csv_file << obj << std::endl;
    csv_file.close();
}

template<typename T>
void write_to_csv(std::vector<T> vector, std::string filename) {
    std::ofstream csv_file;    
    csv_file.open(filename);
    auto iter = vector.begin();
    while(iter != vector.end()) {
        csv_file << *iter;
        ++iter;
        if(iter != vector.end()) csv_file << std::endl;
    }
    csv_file.close();
}

template<typename T>
void write_to_csv(std::vector<T> vector, 
                  std::string filename, 
                  std::string* columns, 
                  std::size_t column_size) {
    std::ofstream csv_file;    
    csv_file.open(filename);
    for(std::size_t index = 0; index < column_size; index++) {
        csv_file << columns[index];
        if(index < column_size - 1) csv_file << ',';
    }
    csv_file << '\n';
    auto iter = vector.begin();
    while(iter != vector.end()) {
        csv_file << *iter;
        ++iter;
        if(iter != vector.end()) csv_file << std::endl;
    }
    csv_file.close();
}
#endif