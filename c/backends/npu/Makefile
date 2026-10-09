CC ?= cc
AR ?= ar
CFLAGS ?= -O2 -g
WARN = -std=c99 -Wall -Wextra -Werror
OUT ?= build
BASE = coli_npu_buf.c coli_hexagon_engine.c

.PHONY: all test clean
all: $(OUT)/libcoli_npu_hexagon.a

$(OUT):
	mkdir -p "$@"

$(OUT)/%.o: %.c coli_npu_buf.h coli_hexagon_engine.h | $(OUT)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARN) -pthread -c $< -o $@

$(OUT)/libcoli_npu_hexagon.a: $(OUT)/coli_npu_buf.o $(OUT)/coli_hexagon_engine.o
	$(AR) rcs $@ $^

$(OUT)/test_hexagon: test_hexagon.c $(BASE) coli_npu_buf.h coli_hexagon_engine.h | $(OUT)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARN) -DCOLI_NPU_TESTING -pthread test_hexagon.c $(BASE) -ldl -lm -o $@

test: $(OUT)/test_hexagon
	./$(OUT)/test_hexagon

clean:
	rm -f $(OUT)/*.o $(OUT)/libcoli_npu_hexagon.a $(OUT)/test_hexagon
