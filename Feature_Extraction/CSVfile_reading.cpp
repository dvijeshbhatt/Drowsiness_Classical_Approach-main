#include<bits/stdc++.h>
#include<tuple>
#include<iostream>
#include<fstream>
#include<sstream>
#include<string>
using namespace std;
  

// Function to print 2D vector elements
void print(vector<tuple<int,int, int> > & myContainer)
{
    // Iterating over 2D vector elements
    for (auto currentVector : myContainer) {
        // Each element of the 2D vector is a vector itself
        tuple<int, int, int> myVector = currentVector;
 
        // Iterating over the vector elements
        cout << "[";
            // Print the element
            cout << get<0>(currentVector) << ", " << get<1>(currentVector) << ", " << get<2>(currentVector);
        cout << "]\n";
    }
}


void readcsvfile(string filename){

    vector<vector<int>> content;
    vector<int> row;
    string line, word;
    
    fstream file("rcam_1_EAR.csv",ios::in);

    if(file.is_open()){
        while(getline(file,line)){
            row.clear();

            istringstream str(line);
            while(getline(str,word,',')){
                row.push_back(stoi(word));
            }
            content.push_back(row);
        }
    }
    for(int i=0;i<content.size();i++){
        for(int j=0;j<content[i].size();j++){
            cout<<(content[i][j])<<" ";   
        }
        cout<<"\n";
    }

    
    /*vector<std::vector<std::string>> csvRows;

    for (std::string line; std::getline(input, line);) {
        std::istringstream ss(std::move(line));
        std::vector<std::string> row;

        //std::getline can split on other characters, here we use ','
        for (std::string value; std::getline(ss, value, ',');) 
        {
            cout<<value<<"\t";
            cout<<typeid(value).name();
            row.push_back(std::move(value));
        }
        cout<<"\n";
        csvRows.push_back(std::move(row));
    }

    // Print out our table
    /*for (const std::vector<std::string>& row : csvRows) {
        for (const std::string& value : row) {
                std::cout << std::setw(15) << stoi(value) << "\t";
        }
        std::cout << "\n";
    }*/

}

// Driver code
int main()
{
    // Declaring a 2D vector of tuples
    vector<tuple<int,int, int>> myContainer;
 
    // Initializing vectors of tuples
    // tuples are of type {int, int, int}
    tuple<int, int, int> vect1 = { 1, 1, 2 };
 
    tuple<int, int, int> vect2 = { 1, 2, 3 };
 
    tuple<int, int, int> vect3 = { 4, 5, 2 };
 
    // Inserting vector of tuples in the 2D vector
    myContainer.push_back(vect1);
    myContainer.push_back(vect2);
    myContainer.push_back(vect3);
 
    // Calling print function
    print(myContainer);

    readcsvfile("rcam_1_EAR.csv");
 
    return 0;
}