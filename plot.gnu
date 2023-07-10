set datafile separator ","
set multiplot layout 1,2

set origin 0,0
set size 0.5,1
splot "u_gnuplot.txt" using 1:2:3

set origin 0.5,0
set size 0.5,1
splot "w_gnuplot.txt" using 1:2:3

unset multiplot

pause -1
