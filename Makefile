#LFLAGS += `pkg-config --libs gsl`
#CXXFLAGS += `pkg-config --cflags gsl`

# -L/media/nikhil/hdv/gsl-build/build/lib
LFLAGS += -L. -lgicp -Lann_1.1.1/lib -lANN -L/media/nikhil/hdv/gsl-build/build/lib -lgsl -lgslcblas \
	  -lboost_program_options -lboost_system -lboost_timer -lstdc++ -lflann_cuda
	  
#-I/media/nikhil/hdv/gsl-build/build/include 
CXXFLAGS += -O3 -I./ann_1.1.1/include/ANN -I/media/nikhil/hdv/gsl-build/build/include -I/usr/local/include

LINK = g++
CXX = g++

SOURCES = optimize.cpp gicp.cpp bfgs_funcs.cpp scan.cpp transform.c scan2ascii.cpp

BINARIES = test_gicp scan2ascii

TARGETS = libgicp.a test_gicp scan2ascii

.SUFFIXES:
.SUFFIXES: .o .c .cpp .a

# rules
all: $(TARGETS)

libgicp.a: gicp.o optimize.o bfgs_funcs.o transform.o scan.o
	ar rvs $@ $^

test_gicp: test_gicp.o gicp.o optimize.o bfgs_funcs.o transform.o

scan2ascii: scan.o scan2ascii.o transform.o

clean:
	rm -f *.o $(TARGETS) *~ t_*.tfm *.dat mahalanobis.txt correspondence* iterations.txt

$(BINARIES):
	$(LINK) -o $@ $^ $(LFLAGS)

.cpp.o: 
	$(CXX) $(CXXFLAGS) -c $< -o $@
