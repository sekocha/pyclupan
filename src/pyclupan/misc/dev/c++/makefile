cpp = g++

cflags = -O3 -fPIC -DNDEBUG -Wno-deprecated -mtune=native -march=native -std=c++11 -fpermissive
ompflags = -fopenmp

include_common = -I./src
include_pybind11 = `python3 -m pybind11 --includes`
include_eigen3 = -I${CONDA_PREFIX}/include/eigen3
INCLUDE = $(include_common) $(include_eigen3) $(include_pybind11)

loader = $(cpp)

libomp = -lgomp
libs = $(libomp)

VPATH = ./src
source = nonequiv_labelings.o correlation.o pybind11_pyclupan.o

all: mkdir $(addprefix lib/,$(source))

	$(loader) $(ldflags) -shared $(addprefix lib/,$(source)) $(libs) \
        -o lib/pyclupancpp.so

mkdir:
	@if [ ! -d lib ]; then \
        echo "mkdir lib"; mkdir lib; \
    fi

lib/%.o: %.cpp
	$(cpp) -c $(cflags) $(ompflags) $(INCLUDE) -o $@ $<

clean:
	\rm -rf lib
