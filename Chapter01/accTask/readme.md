pgc++ -std=c++11 -acc -O3 -ta=multicore,nvidia -Minfo=accel accTask.cpp -o accTask

$ pgc++ -Ofast -std=c++11 accTask.cpp -o accTask.single

pgc++-Error-Unknown switch: -Ofast

pgc++ -fast -std=c++11 accTask.cpp -o accTask.single

$ ./accTask.single 1 10000

Duration 14.5115 second

Final sum is 1

rmfarber@bd:~/OpenACC_book/myChapter$ ./accTask.single 4 10000

Duration 57.4627 second

Final sum is 4