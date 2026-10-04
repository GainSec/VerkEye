#define _GNU_SOURCE
#define _FILE_OFFSET_BITS 64

#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdarg.h>
#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/ioctl.h>
#include <sys/stat.h>
#include <sys/types.h>
#include <unistd.h>

#define CAVALRY_RUN_DAGS UINT32_C(0xc0084303)
#define CAVALRY_GET_AUDIO_CLK UINT32_C(0x80084380)
#define RECOVERED_AUDIO_CLOCK_HZ UINT64_C(12288000)
#define DEFAULT_SNAPSHOT_BYTES ((size_t)65536)
#define MAX_SNAPSHOT_BYTES ((size_t)4194304)
#define TRACE_SCHEMA "verkeye.cv22.ioctl-shim.v1"

typedef int (*ioctl_function)(int, unsigned long, ...);

static ioctl_function next_ioctl;
static unsigned int event_sequence;
static unsigned int run_sequence;

static ioctl_function resolve_next_ioctl(void) {
    union {
        void *object;
        ioctl_function function;
    } symbol;

    if (next_ioctl != NULL) {
        return next_ioctl;
    }
    dlerror();
    symbol.object = dlsym(RTLD_NEXT, "ioctl");
    if (symbol.object == NULL || dlerror() != NULL) {
        return NULL;
    }
    next_ioctl = symbol.function;
    return next_ioctl;
}

static const char *capture_directory(void) {
    const char *directory = getenv("VERKEYE_CAVALRY_CAPTURE_DIR");
    if (directory == NULL || directory[0] == '\0') {
        return NULL;
    }
    return directory;
}

static size_t snapshot_size(void) {
    const char *text = getenv("VERKEYE_CAVALRY_SNAPSHOT_BYTES");
    size_t parsed = 0;
    size_t index;
    unsigned int digit;

    if (text == NULL || text[0] == '\0') {
        return DEFAULT_SNAPSHOT_BYTES;
    }
    for (index = 0; text[index] != '\0'; ++index) {
        if (text[index] < '0' || text[index] > '9') {
            return 0;
        }
        digit = (unsigned int)(text[index] - '0');
        if (parsed > (MAX_SNAPSHOT_BYTES - digit) / 10) {
            return 0;
        }
        parsed = parsed * 10 + digit;
    }
    return parsed == 0 ? 0 : parsed;
}

static uint64_t fnv1a64(const unsigned char *data, size_t size) {
    uint64_t value = UINT64_C(14695981039346656037);
    size_t index;
    for (index = 0; index < size; ++index) {
        value ^= data[index];
        value *= UINT64_C(1099511628211);
    }
    return value;
}

static int trace_fd(void) {
    const char *directory = capture_directory();
    char path[PATH_MAX];
    int length;

    if (directory == NULL) {
        errno = ENOENT;
        return -1;
    }
    length = snprintf(path, sizeof(path), "%s/ioctl.jsonl", directory);
    if (length < 0 || (size_t)length >= sizeof(path)) {
        errno = ENAMETOOLONG;
        return -1;
    }
    return open(path, O_WRONLY | O_CREAT | O_APPEND | O_CLOEXEC, 0600);
}

static void trace_audio_clock(int outcome_errno) {
    int fd = trace_fd();
    if (fd < 0) {
        return;
    }
    (void)dprintf(
        fd,
        "{\"schema\":\"%s\",\"sequence\":%u,"
        "\"request\":\"0x%08x\",\"operation\":"
        "\"CAVALRY_GET_AUDIO_CLK\",\"outcome\":\"%s\","
        "\"audio_clock_hz\":%llu,\"errno\":%d}\n",
        TRACE_SCHEMA, event_sequence++, CAVALRY_GET_AUDIO_CLK,
        outcome_errno == 0 ? "emulated" : "failed",
        (unsigned long long)RECOVERED_AUDIO_CLOCK_HZ, outcome_errno);
    (void)close(fd);
}

static void trace_run_dags(
    const char *outcome,
    const char *snapshot_name,
    size_t requested_bytes,
    size_t captured_bytes,
    uint64_t digest,
    int outcome_errno
) {
    int fd = trace_fd();
    if (fd < 0) {
        return;
    }
    (void)dprintf(
        fd,
        "{\"schema\":\"%s\",\"sequence\":%u,"
        "\"request\":\"0x%08x\",\"operation\":"
        "\"CAVALRY_RUN_DAGS\",\"outcome\":\"%s\","
        "\"snapshot_file\":\"%s\",\"requested_bytes\":%zu,"
        "\"captured_bytes\":%zu,\"fnv1a64\":\"%016llx\","
        "\"errno\":%d}\n",
        TRACE_SCHEMA, event_sequence++, CAVALRY_RUN_DAGS, outcome,
        snapshot_name, requested_bytes, captured_bytes,
        (unsigned long long)digest, outcome_errno);
    (void)close(fd);
}

static ssize_t copy_process_memory(
    const void *source,
    unsigned char *destination,
    size_t size
) {
    int fd;
    ssize_t result;

    fd = open("/proc/self/mem", O_RDONLY | O_CLOEXEC);
    if (fd < 0) {
        return -1;
    }
    result = pread(fd, destination, size, (off_t)(uintptr_t)source);
    (void)close(fd);
    return result;
}

static int write_snapshot(
    const void *argument,
    size_t requested_bytes,
    char *snapshot_name,
    size_t snapshot_name_size,
    size_t *captured_bytes,
    uint64_t *digest
) {
    const char *directory = capture_directory();
    unsigned char *buffer;
    char path[PATH_MAX];
    ssize_t copied;
    ssize_t written;
    int file_descriptor;
    int length;

    *captured_bytes = 0;
    *digest = 0;
    if (directory == NULL || argument == NULL || requested_bytes == 0) {
        errno = argument == NULL ? EFAULT : EINVAL;
        return -1;
    }
    length = snprintf(
        snapshot_name, snapshot_name_size, "run-dags-%06u.bin", run_sequence++
    );
    if (length < 0 || (size_t)length >= snapshot_name_size) {
        errno = ENAMETOOLONG;
        return -1;
    }
    length = snprintf(path, sizeof(path), "%s/%s", directory, snapshot_name);
    if (length < 0 || (size_t)length >= sizeof(path)) {
        errno = ENAMETOOLONG;
        return -1;
    }
    buffer = malloc(requested_bytes);
    if (buffer == NULL) {
        return -1;
    }
    copied = copy_process_memory(argument, buffer, requested_bytes);
    if (copied <= 0) {
        free(buffer);
        return -1;
    }
    file_descriptor = open(
        path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600
    );
    if (file_descriptor < 0) {
        free(buffer);
        return -1;
    }
    written = write(file_descriptor, buffer, (size_t)copied);
    if (written != copied) {
        int saved_errno = errno == 0 ? EIO : errno;
        (void)close(file_descriptor);
        free(buffer);
        errno = saved_errno;
        return -1;
    }
    *captured_bytes = (size_t)copied;
    *digest = fnv1a64(buffer, *captured_bytes);
    (void)close(file_descriptor);
    free(buffer);
    return 0;
}

static int handle_audio_clock(void *argument) {
    if (argument == NULL) {
        trace_audio_clock(EFAULT);
        errno = EFAULT;
        return -1;
    }
    memcpy(argument, &(uint64_t){RECOVERED_AUDIO_CLOCK_HZ}, sizeof(uint64_t));
    trace_audio_clock(0);
    return 0;
}

static int handle_run_dags(void *argument) {
    char snapshot_name[64] = "";
    size_t requested_bytes = snapshot_size();
    size_t captured_bytes = 0;
    uint64_t digest = 0;
    int capture_errno;
    const char *outcome;

    if (write_snapshot(
            argument, requested_bytes, snapshot_name, sizeof(snapshot_name),
            &captured_bytes, &digest
        ) == 0) {
        outcome = "captured_unsupported";
        capture_errno = ENOTSUP;
    } else {
        capture_errno = errno == 0 ? EIO : errno;
        outcome = "capture_failed";
    }
    trace_run_dags(
        outcome, snapshot_name, requested_bytes, captured_bytes, digest,
        capture_errno
    );
    errno = capture_errno == ENOTSUP ? ENOTSUP : capture_errno;
    return -1;
}

int ioctl(int file_descriptor, unsigned long request, ...) {
    ioctl_function real_ioctl;
    void *argument = NULL;
    bool has_argument = _IOC_DIR(request) != _IOC_NONE || _IOC_SIZE(request) != 0;
    va_list arguments;

    if (has_argument) {
        va_start(arguments, request);
        argument = va_arg(arguments, void *);
        va_end(arguments);
    }
    if (request == CAVALRY_GET_AUDIO_CLK) {
        return handle_audio_clock(argument);
    }
    if (request == CAVALRY_RUN_DAGS) {
        return handle_run_dags(argument);
    }
    real_ioctl = resolve_next_ioctl();
    if (real_ioctl == NULL) {
        errno = ENOSYS;
        return -1;
    }
    if (has_argument) {
        return real_ioctl(file_descriptor, request, argument);
    }
    return real_ioctl(file_descriptor, request);
}
