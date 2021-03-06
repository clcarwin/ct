PREFIX=/home/jure/build/torch7
CFLAGS=-I$(PREFIX)/include/THC -I$(PREFIX)/include/TH -I$(PREFIX)/include
LDFLAGS=-L$(PREFIX)/lib -Xlinker -rpath,$(PREFIX)/lib -lcublas -lluaT -lTHC -lTH

libct.so: ct.cu
	nvcc -arch sm_35 --compiler-options '-fPIC' -o libct.so --shared ct.cu $(CFLAGS) $(LDFLAGS)
