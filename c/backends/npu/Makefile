CC ?= cc
AR ?= ar
CFLAGS ?= -O2 -g
WARN = -std=c99 -Wall -Wextra -Werror
OUT ?= build
BASE = coli_npu_buf.c coli_hexagon_engine.c coli_npu_quant.c

.PHONY: all test clean
all: $(OUT)/libcoli_npu_hexagon.a

$(OUT):
	mkdir -p "$@"

$(OUT)/%.o: %.c coli_npu_buf.h coli_hexagon_engine.h | $(OUT)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARN) -pthread -c $< -o $@

$(OUT)/coli_npu_quant.o: coli_npu_quant.h

$(OUT)/libcoli_npu_hexagon.a: $(OUT)/coli_npu_buf.o $(OUT)/coli_hexagon_engine.o $(OUT)/coli_npu_quant.o
	$(AR) rcs $@ $^

$(OUT)/test_hexagon: test_hexagon.c $(BASE) coli_npu_buf.h coli_hexagon_engine.h | $(OUT)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARN) -DCOLI_NPU_TESTING -pthread test_hexagon.c $(BASE) -ldl -lm -o $@

$(OUT)/test_quant: test_quant.c coli_npu_quant.c coli_npu_quant.h | $(OUT)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARN) test_quant.c coli_npu_quant.c -o $@

$(OUT)/quant_scalar.o: coli_npu_quant.c coli_npu_quant.h | $(OUT)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARN) -DCOLI_NPU_QUANT_SCALAR -Dcoli_npu_i4_f16_expand=coli_npu_i4_f16_expand_scalar -c $< -o $@

$(OUT)/test_quant_equivalence: test_quant_equivalence.c coli_npu_quant.c coli_npu_quant.h $(OUT)/quant_scalar.o | $(OUT)
	$(CC) $(CPPFLAGS) $(CFLAGS) $(WARN) test_quant_equivalence.c coli_npu_quant.c $(OUT)/quant_scalar.o -o $@

test: $(OUT)/test_hexagon $(OUT)/test_quant $(OUT)/test_quant_equivalence
	./$(OUT)/test_hexagon
	./$(OUT)/test_quant
	./$(OUT)/test_quant_equivalence

clean:
	rm -f $(OUT)/*.o $(OUT)/libcoli_npu_hexagon.a $(OUT)/test_hexagon $(OUT)/test_quant $(OUT)/test_quant_equivalence
