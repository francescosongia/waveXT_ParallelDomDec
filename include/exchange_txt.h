#ifndef EXCHANGE_TXT_H
#define EXCHANGE_TXT_H

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <fstream>

typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double

/*
HELPER FUNCTIONS TO EXCHANGE INFORMATION BETWEEN TXT FILES AND EIGEN
*/

SpMat readMat_fromtxt(const std::string& filename, unsigned int rows,unsigned int cols) {  
    std::vector<T> tripletList;
    std::vector<std::string> row;
    std::string line, word;

    std::fstream file (filename, std::ios::in);
    if(file.is_open()){
        while(getline(file, line)){
            row.clear();
            std::stringstream str(line);
            while(getline(str, word, ','))
                row.push_back(word);
            tripletList.emplace_back(std::stod(row[0])-1,std::stod(row[1])-1,std::stod(row[2]));
        }
    }
    else
        std::cerr<<"Could not open the file"<<std::endl;

    SpMat A(rows,cols);
    A.setFromTriplets(tripletList.begin(), tripletList.end());
    return A;
}


void saveVec_totxt(const std::string& filename,const Eigen::VectorXd& v) {
    std::ofstream file;
    file.open(filename);
    if (file.is_open()){
        for(Eigen::Index i=0;i<v.size();++i) {
            file << v[i] << '\n';
        }
    }
    else
        std::cerr<<"file not open"<<std::endl;
}

#endif //EXCHANGE_TXT_H
