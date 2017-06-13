cmake_minimum_required(VERSION 3.5)
project(SemiC)

# set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++0")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} -Wall")
set(SOURCE_FILES
        dat/.
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
        src/reader.c
        src/sequence.h
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