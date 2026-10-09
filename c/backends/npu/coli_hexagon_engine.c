#define _GNU_SOURCE
#define _FILE_OFFSET_BITS 64
#include "coli_hexagon_engine.h"
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <pthread.h>
#include <stdlib.h>
#include <sys/stat.h>
#include <unistd.h>

enum SlotState { EMPTY, QUEUED, READING, READY, FAILED, EXECUTING };
typedef struct {
    ColiNpuBuf *buffer;
    ColiHexagonJob job;
    enum SlotState state;
    int error;
} Slot;
struct ColiHexagonEngine {
    Slot slots[2];
    pthread_mutex_t mutex;
    pthread_cond_t changed;
    pthread_t reader;
    int stop, poisoned, fd;
    uint64_t corpus_bytes;
    ColiHexagonOps ops;
    void *user;
};

static int read_job(ColiHexagonEngine *e, Slot *s) {
    void *p;
    size_t done = 0;
    int rc, end_rc;
    rc = coli_npu_buf_begin(s->buffer, 1, &p);
    if (rc) return rc;
    while (done < s->job.bytes) {
        size_t n = s->job.bytes - done;
        ssize_t got;
        if (n > 16 * 1024 * 1024) n = 16 * 1024 * 1024;
        got = pread(e->fd, (char *)p + done, n, (off_t)(s->job.offset + done));
        if (got < 0) { if (errno == EINTR) continue; rc = errno; break; }
        if (!got) { rc = EIO; break; }
        done += (size_t)got;
    }
    end_rc = coli_npu_buf_end(s->buffer);
    return rc ? rc : end_rc;
}
static void *reader_main(void *arg) {
    ColiHexagonEngine *e = arg;
    pthread_mutex_lock(&e->mutex);
    for (;;) {
        unsigned i;
        int rc;
        Slot *s = NULL;
        for (i = 0; i < 2; ++i) if (e->slots[i].state == QUEUED) { s = &e->slots[i]; break; }
        if (e->stop) break;
        if (!s) { pthread_cond_wait(&e->changed, &e->mutex); continue; }
        s->state = READING;
        pthread_mutex_unlock(&e->mutex);
        rc = read_job(e, s);
        pthread_mutex_lock(&e->mutex);
        s->error = rc;
        s->state = rc ? FAILED : READY;
        pthread_cond_broadcast(&e->changed);
    }
    pthread_mutex_unlock(&e->mutex);
    return NULL;
}
int coli_hexagon_engine_create(ColiHexagonEngine **out, int fd,
                              ColiNpuBuf *a, ColiNpuBuf *b,
                              const ColiHexagonOps *ops, void *user) {
    ColiHexagonEngine *e;
    struct stat st;
    int rc;
    if (!out || *out || !a || !b || a == b || !ops || !ops->begin_token ||
        !ops->dense_route || !ops->execute || !ops->finish_layer || !ops->finish_token) return EINVAL;
    if (fstat(fd, &st) < 0) return errno;
    if (!S_ISREG(st.st_mode) || st.st_size <= 0) return EINVAL;
    e = calloc(1, sizeof(*e));
    if (!e) return ENOMEM;
    e->fd = fcntl(fd, F_DUPFD_CLOEXEC, 0);
    if (e->fd < 0) { rc = errno; free(e); return rc; }
    e->corpus_bytes = (uint64_t)st.st_size;
    e->ops = *ops; e->user = user;
    if ((rc = coli_npu_buf_retain(a))) goto fail_fd;
    if ((rc = coli_npu_buf_retain(b))) goto fail_a;
    e->slots[0].buffer = a; e->slots[1].buffer = b;
    if ((rc = pthread_mutex_init(&e->mutex, NULL))) goto fail_b;
    if ((rc = pthread_cond_init(&e->changed, NULL))) goto fail_mutex;
    if ((rc = pthread_create(&e->reader, NULL, reader_main, e))) goto fail_cond;
    *out = e;
    return 0;
fail_cond: pthread_cond_destroy(&e->changed);
fail_mutex: pthread_mutex_destroy(&e->mutex);
fail_b: coli_npu_buf_release(b);
fail_a: coli_npu_buf_release(a);
fail_fd: close(e->fd); free(e); return rc;
}
static int queue(ColiHexagonEngine *e, unsigned slot, const ColiHexagonJob *j) {
    Slot *s = &e->slots[slot];
    pthread_mutex_lock(&e->mutex);
    if (s->state != EMPTY) { pthread_mutex_unlock(&e->mutex); return EBUSY; }
    s->job = *j; s->error = 0; s->state = QUEUED;
    pthread_cond_broadcast(&e->changed);
    pthread_mutex_unlock(&e->mutex);
    return 0;
}
static int run_layer(ColiHexagonEngine *e, uint32_t layer) {
    const ColiHexagonJob *jobs = NULL;
    size_t count = 0, i;
    int rc = e->ops.dense_route(e->user, layer, &jobs, &count);
    if (rc) return rc;
    if (count && !jobs) return EINVAL;
    /* Validate the entire route before any read or execution. No wrapped ranges. */
    for (i = 0; i < count; ++i) {
        const ColiHexagonJob *j = &jobs[i];
        if (!j->bytes || j->bytes > coli_npu_buf_size(e->slots[i % 2].buffer) ||
            j->offset > e->corpus_bytes || j->bytes > e->corpus_bytes - j->offset ||
            j->offset > INT64_MAX || j->bytes > (uint64_t)INT64_MAX - j->offset ||
            !isfinite(j->gate) || j->gate < 0) return EINVAL;
    }
    for (i = 0; i < count && i < 2; ++i) if ((rc = queue(e, (unsigned)i, &jobs[i]))) return rc;
    if (e->ops.dense_tail && (rc = e->ops.dense_tail(e->user, layer))) return rc;
    for (i = 0; i < count; ++i) {
        unsigned slot = (unsigned)(i % 2);
        Slot *s = &e->slots[slot];
        pthread_mutex_lock(&e->mutex);
        while (s->state == QUEUED || s->state == READING)
            pthread_cond_wait(&e->changed, &e->mutex);
        if (s->state != READY) {
            rc = s->error ? s->error : EIO;
            pthread_mutex_unlock(&e->mutex);
            return rc;
        }
        s->state = EXECUTING;
        pthread_mutex_unlock(&e->mutex);
        rc = e->ops.execute(e->user, layer, &jobs[i], s->buffer, slot);
        pthread_mutex_lock(&e->mutex);
        s->state = EMPTY;
        pthread_mutex_unlock(&e->mutex);
        if (rc) return rc;
        if (i + 2 < count && (rc = queue(e, slot, &jobs[i + 2]))) return rc;
    }
    return e->ops.finish_layer(e->user, layer);
}
int coli_hexagon_token_step(ColiHexagonEngine *e, uint64_t position, uint32_t layers) {
    uint32_t layer;
    int rc;
    if (!e || !layers) return EINVAL;
    if (e->poisoned) return ECANCELED;
    rc = e->ops.begin_token(e->user, position);
    for (layer = 0; !rc && layer < layers; ++layer) rc = run_layer(e, layer);
    if (!rc) rc = e->ops.finish_token(e->user);
    /* A partial KV update cannot be retried as if the token had never run. */
    if (rc) e->poisoned = 1;
    return rc;
}
void coli_hexagon_engine_free(ColiHexagonEngine **p) {
    ColiHexagonEngine *e;
    if (!p || !(e = *p)) return;
    pthread_mutex_lock(&e->mutex);
    e->stop = 1;
    pthread_cond_broadcast(&e->changed);
    pthread_mutex_unlock(&e->mutex);
    pthread_join(e->reader, NULL);
    pthread_cond_destroy(&e->changed);
    pthread_mutex_destroy(&e->mutex);
    coli_npu_buf_release(e->slots[0].buffer);
    coli_npu_buf_release(e->slots[1].buffer);
    close(e->fd);
    free(e); *p = NULL;
}
