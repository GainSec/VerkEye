#define _POSIX_C_SOURCE 200809L

#include <dlfcn.h>
#include <errno.h>
#include <fcntl.h>
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define PROBE_SCHEMA "verkeye.cv22.native-probe.v1"

struct nnctrl_net_config {
    const char *filename;
    void *memory;
    uint32_t flags;
    uint32_t priority;
    uint32_t total_dag_bytes;
    uint32_t input_bytes;
    uint32_t output_bytes;
    uint32_t working_bytes;
};

_Static_assert(sizeof(struct nnctrl_net_config) == 40,
               "recovered nnctrl config ABI must remain 40 bytes");
_Static_assert(offsetof(struct nnctrl_net_config, priority) == 20,
               "recovered nnctrl priority offset must remain 20");

struct nnctrl_memory {
    void *virtual_address;
    uint32_t physical_address;
    uint32_t size;
};

_Static_assert(sizeof(struct nnctrl_memory) == 16,
               "recovered nnctrl memory ABI must remain 16 bytes");
_Static_assert(offsetof(struct nnctrl_memory, physical_address) == 8,
               "recovered nnctrl physical-address offset must remain 8");
_Static_assert(offsetof(struct nnctrl_memory, size) == 12,
               "recovered nnctrl size offset must remain 12");

typedef int (*nnctrl_init_fn)(int, unsigned char);
typedef int (*nnctrl_init_net_fn)(struct nnctrl_net_config *, void *, void *);
typedef int (*nnctrl_load_net_fn)(int, struct nnctrl_memory *, void *, void *);
typedef int (*nnctrl_run_net_fn)(int, float *, void *, void *, void *);
typedef int (*nnctrl_exit_net_fn)(int);
typedef void (*nnctrl_exit_fn)(void);
typedef int (*nnctrl_get_version_fn)(void *);

static unsigned int sequence_number;

static void emit_record(const char *operation, int return_code) {
    const char *status = return_code < 0 ? "failed" : "passed";
    printf("VERKEYE_PROBE_JSON "
           "{\"schema\":\"%s\",\"sequence\":%u,\"operation\":\"%s\","
           "\"status\":\"%s\",\"return_code\":%d}\n",
           PROBE_SCHEMA, sequence_number++, operation, status, return_code);
    fflush(stdout);
}

static void emit_config(const struct nnctrl_net_config *config, int net_id) {
    const char *status = net_id < 0 ? "failed" : "passed";
    printf("VERKEYE_PROBE_JSON "
           "{\"schema\":\"%s\",\"sequence\":%u,"
           "\"operation\":\"nnctrl_config\",\"status\":\"%s\","
           "\"return_code\":%d,\"total_dag_bytes\":%u,"
           "\"input_bytes\":%u,\"output_bytes\":%u,"
           "\"working_bytes\":%u}\n",
           PROBE_SCHEMA, sequence_number++, status, net_id,
           config->total_dag_bytes, config->input_bytes, config->output_bytes,
           config->working_bytes);
    fflush(stdout);
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

static void emit_memory(const struct nnctrl_memory *memory, int return_code) {
    const char *status = return_code < 0 ? "failed" : "passed";
    uint64_t digest = fnv1a64(memory->virtual_address, memory->size);
    printf("VERKEYE_PROBE_JSON "
           "{\"schema\":\"%s\",\"sequence\":%u,"
           "\"operation\":\"nnctrl_memory\",\"status\":\"%s\","
           "\"return_code\":%d,\"physical_address\":%u,"
           "\"size\":%u,\"fnv1a64\":\"%016llx\"}\n",
           PROBE_SCHEMA, sequence_number++, status, return_code,
           memory->physical_address, memory->size,
           (unsigned long long)digest);
    fflush(stdout);
}

static void emit_memory_dump(
    const struct nnctrl_memory *memory, int return_code
) {
    const char *status = return_code < 0 ? "failed" : "passed";
    uint64_t digest = fnv1a64(memory->virtual_address, memory->size);
    printf("VERKEYE_PROBE_JSON "
           "{\"schema\":\"%s\",\"sequence\":%u,"
           "\"operation\":\"nnctrl_memory_dump\",\"status\":\"%s\","
           "\"return_code\":%d,\"size\":%u,"
           "\"fnv1a64\":\"%016llx\"}\n",
           PROBE_SCHEMA, sequence_number++, status, return_code, memory->size,
           (unsigned long long)digest);
    fflush(stdout);
}

static int dump_memory(const char *path, const struct nnctrl_memory *memory) {
    const unsigned char *cursor = memory->virtual_address;
    size_t remaining = memory->size;
    int fd;

    fd = open(
        path, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC | O_NOFOLLOW, 0600
    );
    if (fd < 0) {
        return -errno;
    }
    while (remaining > 0) {
        ssize_t written = write(fd, cursor, remaining);
        if (written < 0 && errno == EINTR) {
            continue;
        }
        if (written <= 0) {
            int saved_errno = errno == 0 ? EIO : errno;
            (void)close(fd);
            return -saved_errno;
        }
        cursor += (size_t)written;
        remaining -= (size_t)written;
    }
    if (fsync(fd) < 0) {
        int saved_errno = errno;
        (void)close(fd);
        return -saved_errno;
    }
    if (close(fd) < 0) {
        return -errno;
    }
    return 0;
}

static void *required_symbol(void *library, const char *name) {
    void *symbol;
    dlerror();
    symbol = dlsym(library, name);
    if (symbol == NULL || dlerror() != NULL) {
        fprintf(stderr, "required symbol unavailable: %s\n", name);
        return NULL;
    }
    return symbol;
}

int main(int argc, char **argv) {
    void *library;
    nnctrl_init_fn nnctrl_init;
    nnctrl_init_net_fn nnctrl_init_net;
    nnctrl_load_net_fn nnctrl_load_net;
    nnctrl_run_net_fn nnctrl_run_net;
    nnctrl_exit_net_fn nnctrl_exit_net;
    nnctrl_exit_fn nnctrl_exit;
    nnctrl_get_version_fn nnctrl_get_version;
    struct nnctrl_net_config config;
    struct nnctrl_memory memory;
    unsigned char version[84];
    int fd;
    int rc;
    int net_id;
    int load_rc = -1;
    int dump_rc;
    int run_rc = -1;
    float elapsed_ms = 0.0F;

    if (argc != 4 && argc != 5) {
        fprintf(
            stderr,
            "usage: %s LIBNNCTRL DEVICE MODEL [WORKING_MEMORY_DUMP]\n",
            argv[0]
        );
        return 64;
    }
    dump_rc = argc == 5 ? -1 : 0;

    library = dlopen(argv[1], RTLD_NOW | RTLD_LOCAL);
    emit_record("dlopen", library == NULL ? -1 : 0);
    if (library == NULL) {
        fprintf(stderr, "dlopen failed: %s\n", dlerror());
        return 65;
    }

    nnctrl_init = (nnctrl_init_fn)required_symbol(library, "nnctrl_init");
    nnctrl_init_net =
        (nnctrl_init_net_fn)required_symbol(library, "nnctrl_init_net");
    nnctrl_load_net =
        (nnctrl_load_net_fn)required_symbol(library, "nnctrl_load_net");
    nnctrl_run_net =
        (nnctrl_run_net_fn)required_symbol(library, "nnctrl_run_net");
    nnctrl_exit_net =
        (nnctrl_exit_net_fn)required_symbol(library, "nnctrl_exit_net");
    nnctrl_exit = (nnctrl_exit_fn)required_symbol(library, "nnctrl_exit");
    nnctrl_get_version =
        (nnctrl_get_version_fn)required_symbol(library, "nnctrl_get_version");
    rc = nnctrl_init != NULL && nnctrl_init_net != NULL &&
                 nnctrl_load_net != NULL && nnctrl_run_net != NULL &&
                 nnctrl_exit_net != NULL && nnctrl_exit != NULL &&
                 nnctrl_get_version != NULL
             ? 0
             : -1;
    emit_record("dlsym", rc);
    if (rc < 0) {
        dlclose(library);
        return 66;
    }

    fd = open(argv[2], O_RDWR | O_CLOEXEC);
    emit_record("open_device", fd < 0 ? -errno : fd);
    if (fd < 0) {
        dlclose(library);
        return 67;
    }

    memset(version, 0, sizeof(version));
    rc = nnctrl_get_version(version);
    emit_record("nnctrl_get_version", rc);
    if (rc < 0) {
        close(fd);
        dlclose(library);
        return 68;
    }

    rc = nnctrl_init(fd, 1);
    emit_record("nnctrl_init", rc);
    if (rc < 0) {
        close(fd);
        dlclose(library);
        return 69;
    }

    memset(&config, 0, sizeof(config));
    memset(&memory, 0, sizeof(memory));
    config.filename = argv[3];
    net_id = nnctrl_init_net(&config, NULL, NULL);
    emit_record("nnctrl_init_net", net_id);
    emit_config(&config, net_id);

    if (net_id >= 0) {
        memory.size = config.working_bytes;
        memory.physical_address = UINT32_C(0x10000000);
        rc = posix_memalign(&memory.virtual_address, 4096, memory.size);
        if (rc != 0) {
            emit_record("allocate_working_memory", -rc);
        } else {
            memset(memory.virtual_address, 0, memory.size);
            emit_record("allocate_working_memory", 0);
            rc = nnctrl_load_net(net_id, &memory, NULL, NULL);
            load_rc = rc;
            emit_record("nnctrl_load_net", rc);
            emit_memory(&memory, rc);
            if (rc >= 0) {
                if (argc == 5) {
                    dump_rc = dump_memory(argv[4], &memory);
                    emit_memory_dump(&memory, dump_rc);
                }
            }
            if (rc >= 0 && dump_rc >= 0) {
                run_rc = nnctrl_run_net(net_id, &elapsed_ms, NULL, NULL, NULL);
                emit_record("nnctrl_run_net", run_rc);
            }
        }
        rc = nnctrl_exit_net(net_id);
        emit_record("nnctrl_exit_net", rc);
    }
    nnctrl_exit();
    emit_record("nnctrl_exit", 0);
    free(memory.virtual_address);
    close(fd);
    dlclose(library);
    return net_id < 0 || load_rc < 0 || dump_rc < 0 || run_rc < 0 ? 70 : 0;
}
