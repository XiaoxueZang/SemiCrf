# SemiCrf
SemiCrf in C

Usage:
$ make 
$ ./wapiti train dat/train_small.txt dat/model_small   (train with data of small size)

$ ./wapiti train dat/train_large.txt dat/model_large  (train with data of large size. 5224 sequences)

./wapiti label -m dat/model_small dat/test_small.txt dat/output.txt
(label the data)

-1 (num): change the rho1 (default 0).
-2 (num): change the rho2 (default 0.0001).
-t (num): determine the number of threads (default 1).
-e (num): stop epsilon.

Example:
./wapiti train -1 0 -t 2 dat/train_small.txt dat/model_small (test with data of small size and run in 2 threads).

