cmake_minimum_required(VERSION 3.5)
project(SemiC)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
set(SOURCE_FILES
        dat/chpattern.txt
        dat/chtest.txt
        dat/chtrain.txt
        dat/model
        dat/model_temp
        dat/nppattern.txt
        dat/nptest.txt
        dat/nptrain.txt
        dat/pattern.txt
        dat/pattern_temp.txt
        dat/test.txt
        dat/train.txt
        dat/train_temp.txt
        src/features.c
        src/features.h
        src/model.c
        src/model.h
        src/options.c
        src/options.h
        src/pattern.c
        src/pattern.h
        src/progress.c
        src/progress.h
        src/quark.c
        src/quark.h
        src/reader.c
        src/reader.h
        # src/rprop.c
        src/sequence.h
        # src/sgdl1.c
        src/thread.c
        src/thread.h
        src/tools.c
        src/tools.h
        src/vmath.c
        src/vmath.h
        src/lbfgs.c
        src/gradient.h
        src/gradient.c
        src/decoder.c
        src/decoder.h
        src/featuresUnitTest.c
        )

add_executable(SemiC ${SOURCE_FILES})