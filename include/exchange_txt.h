#ifndef EXCHANGE_TXT_H
#define EXCHANGE_TXT_H

#include <Eigen/Sparse>
#include <Eigen/Dense>
#include <fstream>

typedef Eigen::SparseMatrix<double> SpMat; // declares a column-major sparse matrix type of double
//typedef Eigen::SparseMatrix<double,Eigen::RowMajor> SpMat;

SpMat readMat_fromtxt(const std::string& filename, unsigned int rows,unsigned int cols) {  //dim:nln*nt*nx*2
    std::vector<T> tripletList;
    //tripletList.reserve();

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
        std::cout<<"Could not open the file"<<std::endl;

    SpMat A(rows,cols);
    A.setFromTriplets(tripletList.begin(), tripletList.end());
    return A;
}

//Eigen::VectorXd readVec_fromtxt(const std::string& filename);

//void saveSMat_totxt(const std::string& filename,const SpMat& A);

void saveVec_totxt(const std::string& filename,const Eigen::VectorXd& v) {
    std::ofstream file;
    file.open(filename);
    if (file.is_open()){
        for(Eigen::Index i=0;i<v.size();++i) {
            file << v[i] << '\n';
        }
    }
    else
        std::cout<<"file not open"<<std::endl;
}

#endif //EXCHANGE_TXT_H
