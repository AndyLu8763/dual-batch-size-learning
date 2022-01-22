set term pngcairo transparent
set output 'acc_MUfactor.png'
set datafile separator ','
set xlabel 'Epoch'
set ylabel 'Testing Accuracy'
#set title 'Accuracy of Training With/without Model-Update Factor, k=1.1'
set key bottom right Left
set xrange [-5:145]

plot './csv/acc_MUfactor.csv' using 1:2 with lines title '1 small, B_S=38, use', \
     './csv/acc_MUfactor.csv' using 1:3 with lines title '1 small, B_S=38, -' dashtype 2, \
     './csv/acc_MUfactor.csv' using 1:4 with lines title '2 small, B_S=87, use', \
     './csv/acc_MUfactor.csv' using 1:5 with lines title '2 small, B_S=87, -' dashtype 2, \
     './csv/acc_MUfactor.csv' using 1:6 with lines title '3 small, B_S=127, use', \
     './csv/acc_MUfactor.csv' using 1:7 with lines title '3 small, B_S=127, -' dashtype 2
