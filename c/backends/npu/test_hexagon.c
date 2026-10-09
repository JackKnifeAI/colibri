#define _GNU_SOURCE
#include "coli_hexagon_engine.h"
#include <assert.h>
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

typedef struct {
    ColiHexagonJob jobs[7];
    size_t count;
    unsigned calls, finished, tails;
    int fail_execute, transform, fail_prepare;
} Test;
static int begin(void *u, uint64_t p) { (void)u; return p == 42 ? 0 : EINVAL; }
static int route(void *u, uint32_t l, const ColiHexagonJob **j, size_t *n) {
    Test *t = u; (void)l; *j = t->jobs; *n = t->count; return 0;
}
static int tail(void *u, uint32_t l) { Test *t = u; (void)l; ++t->tails; return 0; }
static int execute(void *u, uint32_t l, const ColiHexagonJob *j, ColiNpuBuf *b, unsigned slot) {
    Test *t = u;
    unsigned char *p;
    size_t i;
    (void)l;
    assert(j->expert == t->calls % 7);
    assert(slot == j->expert % 2);
    assert(coli_npu_buf_begin(b, 0, (void **)&p) == 0);
    for (i = 0; i < j->bytes; ++i) assert(p[i] == (unsigned char)(j->expert+(t->transform?17:0)));
    assert(coli_npu_buf_begin(b, 1, (void **)&p) == EBUSY);
    assert(coli_npu_buf_end(b) == 0);
    ++t->calls;
    return t->fail_execute ? EIO : 0;
}
static int finish_layer(void *u, uint32_t l) { Test *t = u; (void)l; ++t->finished; return 0; }
static int finish_token(void *u) { (void)u; return 0; }
static int prepare(void *u, const ColiHexagonJob *j, void *mapped, size_t capacity) {
    const Test *t=u;
    unsigned char *p=mapped;
    size_t i;
    assert(capacity>=j->bytes);
    if(t->fail_prepare) return EDOM;
    for(i=0;i<j->bytes;++i) p[i]=(unsigned char)(p[i]+17);
    return 0;
}
int main(void) {
    Test t;
    ColiNpuBuf *a = NULL, *b = NULL;
    ColiHexagonEngine *e = NULL;
    ColiHexagonOps ops = {begin, route, tail, execute, finish_layer, finish_token, NULL};
    char name[4096];
    const char *tmp = getenv("TMPDIR");
    unsigned char data[4096];
    unsigned i;
    int fd;
    assert(snprintf(name, sizeof(name), "%s/coli-hexagon-XXXXXX", tmp ? tmp : "/tmp") < (int)sizeof(name));
    fd = mkstemp(name);
    assert(fd >= 0); unlink(name);
    memset(&t, 0, sizeof(t));
    for (i = 0; i < 7; ++i) {
        memset(data, (int)i, sizeof(data));
        assert(write(fd, data, sizeof(data)) == (ssize_t)sizeof(data));
        t.jobs[i].offset = i * sizeof(data); t.jobs[i].bytes = sizeof(data);
        t.jobs[i].expert = i; t.jobs[i].gate = 1.0f / 7;
    }
    t.count = 7;
    assert(coli_npu_buf_test(&a, sizeof(data)) == 0);
    assert(coli_npu_buf_test(&b, sizeof(data)) == 0);
    assert(coli_hexagon_engine_create(&e, fd, a, a, &ops, &t) == EINVAL);
    assert(coli_hexagon_engine_create(&e, fd, a, b, &ops, &t) == 0);
    assert(coli_npu_buf_free(&a) == EBUSY);
    assert(coli_hexagon_token_step(e, 42, 3) == 0);
    assert(t.calls == 21 && t.finished == 3 && t.tails == 3);
    assert(coli_hexagon_token_step(e, 42, 1) == 0);
    coli_hexagon_engine_free(&e);
    /* Transform executes under the reader's write bracket before foreground use. */
    t.calls=0; t.transform=1; ops.prepare_weights=prepare;
    assert(coli_hexagon_engine_create(&e,fd,a,b,&ops,&t)==0);
    assert(coli_hexagon_token_step(e,42,1)==0);
    assert(t.calls==7);
    coli_hexagon_engine_free(&e);
    t.calls=0; t.fail_prepare=1;
    assert(coli_hexagon_engine_create(&e,fd,a,b,&ops,&t)==0);
    assert(coli_hexagon_token_step(e,42,1)==EDOM);
    assert(t.calls==0);
    assert(coli_hexagon_token_step(e,42,1)==ECANCELED);
    coli_hexagon_engine_free(&e);
    t.transform=0; t.fail_prepare=0; ops.prepare_weights=NULL;
    /* Validate all jobs before dispatching even the first valid one. */
    t.calls = 0; t.jobs[6].offset = UINT64_MAX;
    assert(coli_hexagon_engine_create(&e, fd, a, b, &ops, &t) == 0);
    assert(coli_hexagon_token_step(e, 42, 1) == EINVAL);
    assert(t.calls == 0);
    assert(coli_hexagon_token_step(e, 42, 1) == ECANCELED);
    coli_hexagon_engine_free(&e);
    t.jobs[6].offset = 6 * sizeof(data);
    t.fail_execute = 1;
    assert(coli_hexagon_engine_create(&e, fd, a, b, &ops, &t) == 0);
    assert(coli_hexagon_token_step(e, 42, 1) == EIO);
    assert(t.calls == 1);
    assert(coli_hexagon_token_step(e, 42, 1) == ECANCELED);
    coli_hexagon_engine_free(&e);
    /* Truncate after open: a short read must not execute stale buffer bytes. */
    t.fail_execute = 0; t.calls = 0;
    assert(coli_hexagon_engine_create(&e, fd, a, b, &ops, &t) == 0);
    assert(ftruncate(fd, 17) == 0);
    assert(coli_hexagon_token_step(e, 42, 1) == EIO);
    assert(t.calls == 0);
    coli_hexagon_engine_free(&e);
    close(fd);
    assert(coli_npu_buf_free(&a) == 0);
    assert(coli_npu_buf_free(&b) == 0);
    coli_hexagon_engine_free(&e);
    puts("hexagon scheduler: ordering, bounds, failures, lifetime, short reads PASS");
    return 0;
}
