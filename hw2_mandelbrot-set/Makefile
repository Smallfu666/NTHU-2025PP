# ===== Compilers =====
CC      ?= gcc
CXX     ?= g++
MPICC   ?= mpicc
MPICXX  ?= mpicxx

# ===== libpng via pkg-config (fallback to -lpng -lz) =====
PNG_CFLAGS := $(shell pkg-config --cflags libpng 2>/dev/null)
PNG_LIBS   := $(shell pkg-config --libs   libpng 2>/dev/null)
ifeq ($(strip $(PNG_LIBS)),)
  PNG_LIBS := -lpng -lz
endif

# ===== Common flags =====
COMMON_CFLAGS  := -O3 -msse2 -DNDEBUG
COMMON_LDFLAGS :=
COMMON_LDLIBS  := $(PNG_LIBS) -lm

# pthread 版本需要 -pthread（編譯與連結都要）
PTHREAD_FLAGS  := -pthread

# OpenMP 版本需要 -fopenmp（編譯與連結都要）
OMP_FLAGS      := -fopenmp

# ===== Targets =====
TARGETS := hw2seq hw2a hw2b hw2a_time hw2b_time

.PHONY: all clean
all: $(TARGETS)

# ---- hw2seq (若有單執行緒版本 hw2seq.cc) ----
hw2seq: hw2seq.cc
	$(CXX) $(COMMON_CFLAGS) $(PNG_CFLAGS) -o $@ $< $(COMMON_LDLIBS)

# ---- hw2a (pthread) ----
hw2a: hw2a.cc
	$(CXX) $(COMMON_CFLAGS) $(PTHREAD_FLAGS) $(PNG_CFLAGS) -o $@ $< $(PTHREAD_FLAGS) $(COMMON_LDLIBS)

# ---- hw2b (MPI+OpenMP) ----
hw2b: hw2b.cc
	$(MPICXX) $(COMMON_CFLAGS) $(OMP_FLAGS) $(PNG_CFLAGS) -o $@ $< $(OMP_FLAGS) $(COMMON_LDLIBS)

# ---- hw2a_time (pthread，只有計算計時版) ----
hw2a_time: hw2a_time.cc
	$(CXX) $(COMMON_CFLAGS) $(PTHREAD_FLAGS) $(PNG_CFLAGS) -o $@ $< $(PTHREAD_FLAGS) $(COMMON_LDLIBS)

# ---- hw2b_time (MPI+OpenMP，分段計時版) ----
hw2b_time: hw2b_time.cc
	$(MPICXX) $(COMMON_CFLAGS) $(OMP_FLAGS) $(PNG_CFLAGS) -o $@ $< $(OMP_FLAGS) $(COMMON_LDLIBS)

clean:
	rm -f $(TARGETS) *.o *.png *.txt
