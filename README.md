# SemiCrf
SemiCrf in C

Usage:
$ make 
$ ./wapiti train dat/train_temp.txt dat/model_temp   (test with data of small size)

$ ./wapiti train dat/train_semi.txt dat/model_semi   (test with data of large size. 5224 sequences)

-1 (num): change the rho1 (default 0).
-2 (num): change the rho2 (default 0.0001).
-t (num): determine the number of threads (default 1).

Example:
./wapiti train -1 0 -t 2 dat/train_semi.txt dat/model_semi (test with data of large size and run in 2 threads).