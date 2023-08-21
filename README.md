-------------------------------------------------------------------------------
Joint project for the university courses Advanced Programming for
Scientific Computing (APSC) and Numerical Analysis for Partial
Differential Equations (NAPDE), Politecnico di Milano, a.y. 2022-2023

Title: Parallelization techniques for domain decomposition methods 
       for the wave equation

Authors: Francesco Songia, Enrico Zardi

Mailto: francesco.songia@mail.polimi.it, enrico.zardi@mail.polimi.it

-------------------------------------------------------------------------------
The repo contains 5 directories:
 - doc, containing documentation of the code and of the adopted
   mathematical methodology
 - include, containing header files which have to been included,
   some of them contains the full implementation since they are class templates
 - src, containing the other definitions of the remaining methods 
 - tests, is the folder in which problem instances are saved. It contains
   subfolders including the matrix A, the vector b, the data parameter file 
   and the physical coordinates of the d.o.f. used in the postprocessing plot.
   Multiple instances of the problem can be saved and selected to be launched. 
 - results,  collect the text files output solution of the program and their 
   version ready for a Gnuplot postprocessing. 

-------------------------------------------------------------------------------
During the implementation of this program we have used the following libraries:
 - Eigen
 - GetPot
 - Gnuplot
 - mpi

We have used COMPILATORE to compile the program

-------------------------------------------------------------------------------
-------------------------------------------------------------------------------

TO COMPILE AND TEST THE PROGRAM IT'S ENOUGH TO: 
 - define the Eigen path in Makefile
 - type make help to see available commands
 - type for example make run 2 test1, to run the code with two cores

while in the root folder.

-------------------------------------------------------------------------------
-------------------------------------------------------------------------------
This program works with an already created DG discretized problem from the
NAPDE project. The problem is already represented with the matrix A and the 
right-hand side b, those are stored in .txt files in the tests folders.
Here we report the repo of the NAPDE project
LINK NAPDE REPO
In this repo it is possible to assemble the DG problem matrices using
the create_matrices_PACS_project.m file.

Many parameters must be provided through a data file using GetPot library, 
in the doc/report.pdf there is an example and a description of all of them. 

To generates a plot with Gnuplot of the solution (both displacement and 
velocity) it is needed to run the following in the terminal:
gnuplot plot.gnu

-------------------------------------------------------------------------------
For a complete documentation and description of the code go to the report.pdf in 
the doc folder.









