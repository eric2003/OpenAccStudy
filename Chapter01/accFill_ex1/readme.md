 pgc++ -std=c++11 -acc -ta=multicore,tesla -Minfo=accel accFill_ex1.cpp -o accFill_ex1
 pgc++ -v -std=c++11 -acc -ta=multicore,tesla -Minfo=accel accFill_ex1.cpp -o accFill_ex1
 
$ export ACC_DEVICE_TYPE=host
$ $ ./accFill_ex1 1000
Final sum is 1000 millions
$ export ACC_DEVICE_TYPE=nvidia
$ ./accFill_ex1 1000
Final sum is 1000 millions