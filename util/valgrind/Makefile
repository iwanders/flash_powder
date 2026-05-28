SHELL=/bin/bash


# For torch devel build:
#CPPFLAGS=-g -O2 -I$(VIRTUAL_ENV)/../build/ -I$(VIRTUAL_ENV)/../
#LDLIBS=-L$(VIRTUAL_ENV)/../build/lib/ -Wl,-rpath,$(VIRTUAL_ENV)/../build/lib/ -Wl,--no-as-needed -ltorch -ltorch_cpu

# For virtualenv;
CPPFLAGS=-g -O2 -I$(VIRTUAL_ENV)/lib/python3.13/site-packages/torch/include/ -I$(VIRTUAL_ENV)/../
LDLIBS=-L$(VIRTUAL_ENV)/lib/python3.13/site-packages/torch/lib/ -Wl,-rpath,$(VIRTUAL_ENV)/lib/python3.13/site-packages/torch/lib/ -Wl,--no-as-needed -ltorch -ltorch_cpu

#println!("cargo:rustc-link-arg=-Wl,--no-as-needed");
#println!("cargo:rustc-link-arg=-ltorch");
#// println!("cargo:rustc-link-arg=-L{}", lib_path);
#//
#println!("cargo:rustc-link-arg=-Wl,-rpath,{}", lib_path);


process_suppressions: logged_suppressions.supp
	./wrangle_suppressions.py extract logged_suppressions.supp --output ./combined.supp

main: main.cpp
	g++ $(CPPFLAGS) -o main main.cpp $(LDLIBS)

run: main
	./main


logged_suppressions.supp: main
	#yes | valgrind  --gen-suppressions=yes   ./main
	#--log-file=/tmp/generated_suppressions.supp
	valgrind --gen-suppressions=all  ./main &>  logged_suppressions.supp
