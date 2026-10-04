#include <cstdint>
#include <cstring>
#include <dlfcn.h>
#include <algorithm>
#include <array>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <limits>
#include <set>
#include <sstream>
#include <stdexcept>
#include <string>
#include <sys/stat.h>
#include <vector>

namespace {

template <typename Fn>
Fn symbol(void *handle, const char *name) {
    dlerror();
    auto *address = dlsym(handle, name);
    if (const char *error = dlerror()) {
        throw std::runtime_error(std::string("dlsym(") + name + "): " + error);
    }
    return reinterpret_cast<Fn>(address);
}

std::vector<std::uint32_t> read_words(const std::string &path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        throw std::runtime_error("cannot open " + path);
    }
    const auto size = static_cast<std::size_t>(stream.tellg());
    if (size % sizeof(std::uint32_t) != 0) {
        throw std::runtime_error("DAG binary size is not word aligned");
    }
    stream.seekg(0);
    std::vector<std::uint32_t> words(size / sizeof(std::uint32_t));
    stream.read(reinterpret_cast<char *>(words.data()), static_cast<std::streamsize>(size));
    if (!stream) {
        throw std::runtime_error("short read from " + path);
    }
    return words;
}

std::vector<unsigned char> read_bytes(const std::string &path) {
    std::ifstream stream(path, std::ios::binary | std::ios::ate);
    if (!stream) {
        throw std::runtime_error("cannot open " + path);
    }
    const auto size = static_cast<std::size_t>(stream.tellg());
    stream.seekg(0);
    std::vector<unsigned char> bytes(size);
    stream.read(reinterpret_cast<char *>(bytes.data()),
                static_cast<std::streamsize>(size));
    if (!stream) {
        throw std::runtime_error("short read from " + path);
    }
    return bytes;
}

void write_bytes(const std::string &path, const unsigned char *data,
                 std::size_t size) {
    std::ofstream stream(path, std::ios::binary | std::ios::trunc);
    if (!stream) {
        throw std::runtime_error("cannot create " + path);
    }
    stream.write(reinterpret_cast<const char *>(data),
                 static_cast<std::streamsize>(size));
    if (!stream) {
        throw std::runtime_error("short write to " + path);
    }
}

struct LinearVmemConfig {
    std::uint32_t word_address;
    std::uint32_t word_count;
    std::uint32_t byte_mask;
};

static_assert(sizeof(LinearVmemConfig) == 12);

struct LinearDramConfig {
    std::uint32_t byte_address;
    std::uint32_t byte_count;
};

static_assert(sizeof(LinearDramConfig) == 8);

struct VectorDramConfig {
    std::array<unsigned char, 4> packed;
    std::uint32_t byte_address;
    std::uint32_t pitch_bytes;
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t depth;
    std::uint32_t planes;
};

static_assert(sizeof(VectorDramConfig) == 28);

struct VectorVmemConfig {
    std::array<unsigned char, 4> packed;
    std::uint32_t word_address;
    std::uint32_t width;
    std::uint32_t height;
    std::uint32_t depth;
    std::uint32_t planes;
    std::uint32_t partial_size;
};

static_assert(sizeof(VectorVmemConfig) == 28);

using VectorSetter = void (*)(const int *, std::uint32_t, void *);
using VectorGetter = std::uint32_t (*)(const int *, void *);

VectorSetter vector_setter{};
VectorGetter vector_getter{};

struct BoundedMemoryBackend {
    void **vtable;
    std::vector<unsigned char> vmem;
    std::vector<unsigned char> dram;
    std::size_t loads{};
    std::size_t stores{};
};

std::size_t vmem_element_size_bytes(
    const std::array<unsigned char, 4> &packed) {
    const auto size_code = static_cast<unsigned>(packed[0] & 0x7fU);
    const auto layout_code = static_cast<unsigned>(packed[1] & 0x7fU);
    if (size_code > 2 || layout_code != 0) {
        throw std::runtime_error(
            "unsupported vector VMEM element/layout encoding");
    }
    return std::size_t{1} << size_code;
}

std::size_t dram_element_size_bytes(
    const std::array<unsigned char, 4> &packed) {
    const auto flags = static_cast<unsigned>(packed[0] & 0x7fU);
    const auto format = static_cast<unsigned>(packed[1] & 0x7fU);
    if (flags == 0 && format == 0) {
        return 1;
    }
    if (flags == 0 && format == 1) {
        return 4;
    }
    throw std::runtime_error(
        "unsupported vector DRAM element/layout encoding");
}

std::uint32_t load_little_endian(const std::vector<unsigned char> &memory,
                                 std::size_t offset, std::size_t size) {
    if (offset > memory.size() || size > memory.size() - offset) {
        throw std::runtime_error("vector load exceeds bounded memory");
    }
    std::uint32_t value = 0;
    for (std::size_t index = 0; index < size; ++index) {
        value |= static_cast<std::uint32_t>(memory[offset + index]) <<
                 (index * 8U);
    }
    return value;
}

void store_little_endian(std::vector<unsigned char> &memory,
                         std::size_t offset, std::size_t size,
                         std::uint32_t value) {
    if (offset > memory.size() || size > memory.size() - offset) {
        throw std::runtime_error("vector store exceeds bounded memory");
    }
    for (std::size_t index = 0; index < size; ++index) {
        memory[offset + index] = static_cast<unsigned char>(
            value >> (index * 8U));
    }
}

std::size_t checked_vector_dram_offset(
    const BoundedMemoryBackend &backend, const VectorDramConfig &config,
    std::uint32_t width, std::uint32_t height, std::uint32_t depth,
    std::uint32_t plane) {
    const auto element_size = dram_element_size_bytes(config.packed);
    const std::uint64_t rows_per_plane =
        static_cast<std::uint64_t>(config.height) * config.depth;
    const std::uint64_t row =
        static_cast<std::uint64_t>(plane) * rows_per_plane +
        static_cast<std::uint64_t>(depth) * config.height + height;
    const std::uint64_t offset =
        static_cast<std::uint64_t>(config.byte_address) +
        row * config.pitch_bytes +
        static_cast<std::uint64_t>(width) * element_size;
    if (offset > backend.dram.size() ||
        element_size > backend.dram.size() - offset) {
        throw std::runtime_error(
            "vector DRAM access is outside the bounded in-memory backend");
    }
    return static_cast<std::size_t>(offset);
}

void vector_dram_load(BoundedMemoryBackend *backend,
                      const VectorDramConfig *config, void *callback) {
    if (!config || !callback || !vector_setter) {
        throw std::runtime_error("invalid vector DRAM load arguments");
    }
    std::cerr << "backend.vector_dram_load byte_address=0x" << std::hex
              << config->byte_address << " packed="
              << static_cast<unsigned>(config->packed[0]) << ":"
              << static_cast<unsigned>(config->packed[1]) << ":"
              << static_cast<unsigned>(config->packed[2]) << ":"
              << static_cast<unsigned>(config->packed[3])
              << std::dec << " shape="
              << config->width << "x" << config->height << "x"
              << config->depth << "x" << config->planes
              << " pitch=" << config->pitch_bytes << "\n";
    for (std::uint32_t plane = 0; plane < config->planes; ++plane) {
        for (std::uint32_t depth = 0; depth < config->depth; ++depth) {
            for (std::uint32_t height = 0; height < config->height; ++height) {
                for (std::uint32_t width = 0; width < config->width; ++width) {
                    const int coordinate[]{
                        static_cast<int>(width), static_cast<int>(height),
                        static_cast<int>(depth), static_cast<int>(plane)};
                    const auto offset = checked_vector_dram_offset(
                        *backend, *config, width, height, depth, plane);
                    const auto value = load_little_endian(
                        backend->dram, offset,
                        dram_element_size_bytes(config->packed));
                    vector_setter(coordinate, value, callback);
                }
            }
        }
    }
    ++backend->loads;
}

void vector_dram_store(BoundedMemoryBackend *backend,
                       const VectorDramConfig *config, void *callback) {
    if (!config || !callback || !vector_getter) {
        throw std::runtime_error("invalid vector DRAM store arguments");
    }
    std::cerr << "backend.vector_dram_store byte_address=0x" << std::hex
              << config->byte_address << " packed="
              << static_cast<unsigned>(config->packed[0]) << ":"
              << static_cast<unsigned>(config->packed[1]) << ":"
              << static_cast<unsigned>(config->packed[2]) << ":"
              << static_cast<unsigned>(config->packed[3])
              << std::dec << " shape="
              << config->width << "x" << config->height << "x"
              << config->depth << "x" << config->planes
              << " pitch=" << config->pitch_bytes << "\n";
    for (std::uint32_t plane = 0; plane < config->planes; ++plane) {
        for (std::uint32_t depth = 0; depth < config->depth; ++depth) {
            for (std::uint32_t height = 0; height < config->height; ++height) {
                for (std::uint32_t width = 0; width < config->width; ++width) {
                    const int coordinate[]{
                        static_cast<int>(width), static_cast<int>(height),
                        static_cast<int>(depth), static_cast<int>(plane)};
                    const auto offset = checked_vector_dram_offset(
                        *backend, *config, width, height, depth, plane);
                    store_little_endian(
                        backend->dram, offset,
                        dram_element_size_bytes(config->packed),
                        vector_getter(coordinate, callback));
                }
            }
        }
    }
    ++backend->stores;
}

std::size_t checked_vector_vmem_offset(
    const BoundedMemoryBackend &backend, const VectorVmemConfig &config,
    std::uint32_t width, std::uint32_t height, std::uint32_t depth,
    std::uint32_t plane) {
    const auto element_size = vmem_element_size_bytes(config.packed);
    const std::uint64_t linear =
        (((static_cast<std::uint64_t>(plane) * config.depth + depth) *
          config.height + height) * config.width + width);
    const std::uint64_t offset =
        static_cast<std::uint64_t>(config.word_address) *
            sizeof(std::uint32_t) +
        linear * element_size;
    if (offset > backend.vmem.size() ||
        element_size > backend.vmem.size() - offset) {
        throw std::runtime_error(
            "vector VMEM access is outside the bounded in-memory backend");
    }
    return static_cast<std::size_t>(offset);
}

void vector_vmem_load(BoundedMemoryBackend *backend,
                      const VectorVmemConfig *config, void *callback) {
    if (!config || !callback || !vector_setter) {
        throw std::runtime_error("invalid vector VMEM load arguments");
    }
    std::cerr << "backend.vector_vmem_load word_address=0x" << std::hex
              << config->word_address << " packed="
              << static_cast<unsigned>(config->packed[0]) << ":"
              << static_cast<unsigned>(config->packed[1]) << ":"
              << static_cast<unsigned>(config->packed[2]) << ":"
              << static_cast<unsigned>(config->packed[3])
              << std::dec << " shape="
              << config->width << "x" << config->height << "x"
              << config->depth << "x" << config->planes
              << " partial_size=" << config->partial_size << "\n";
    for (std::uint32_t plane = 0; plane < config->planes; ++plane) {
        for (std::uint32_t depth = 0; depth < config->depth; ++depth) {
            for (std::uint32_t height = 0; height < config->height; ++height) {
                for (std::uint32_t width = 0; width < config->width; ++width) {
                    const int coordinate[]{
                        static_cast<int>(width), static_cast<int>(height),
                        static_cast<int>(depth), static_cast<int>(plane)};
                    const auto offset = checked_vector_vmem_offset(
                        *backend, *config, width, height, depth, plane);
                    const auto value = load_little_endian(
                        backend->vmem, offset,
                        vmem_element_size_bytes(config->packed));
                    vector_setter(coordinate, value, callback);
                }
            }
        }
    }
    ++backend->loads;
}

void vector_vmem_store(BoundedMemoryBackend *backend,
                       const VectorVmemConfig *config, void *callback) {
    if (!config || !callback || !vector_getter) {
        throw std::runtime_error("invalid vector VMEM store arguments");
    }
    std::cerr << "backend.vector_vmem_store word_address=0x" << std::hex
              << config->word_address << " packed="
              << static_cast<unsigned>(config->packed[0]) << ":"
              << static_cast<unsigned>(config->packed[1]) << ":"
              << static_cast<unsigned>(config->packed[2]) << ":"
              << static_cast<unsigned>(config->packed[3])
              << std::dec << " shape="
              << config->width << "x" << config->height << "x"
              << config->depth << "x" << config->planes
              << " partial_size=" << config->partial_size << "\n";
    for (std::uint32_t plane = 0; plane < config->planes; ++plane) {
        for (std::uint32_t depth = 0; depth < config->depth; ++depth) {
            for (std::uint32_t height = 0; height < config->height; ++height) {
                for (std::uint32_t width = 0; width < config->width; ++width) {
                    const int coordinate[]{
                        static_cast<int>(width), static_cast<int>(height),
                        static_cast<int>(depth), static_cast<int>(plane)};
                    const auto offset = checked_vector_vmem_offset(
                        *backend, *config, width, height, depth, plane);
                    store_little_endian(
                        backend->vmem, offset,
                        vmem_element_size_bytes(config->packed),
                        vector_getter(coordinate, callback));
                }
            }
        }
    }
    ++backend->stores;
}

void backend_destructor(BoundedMemoryBackend *) {}

std::pair<std::size_t, std::size_t> checked_vmem_range(
    const BoundedMemoryBackend &backend, const LinearVmemConfig &config) {
    constexpr std::size_t bytes_per_word = sizeof(std::uint32_t);
    if (config.word_address >
        std::numeric_limits<std::size_t>::max() / bytes_per_word ||
        config.word_count >
        std::numeric_limits<std::size_t>::max() / bytes_per_word) {
        throw std::runtime_error("VMEM descriptor overflows host size_t");
    }
    const std::size_t offset =
        static_cast<std::size_t>(config.word_address) * bytes_per_word;
    const std::size_t size =
        static_cast<std::size_t>(config.word_count) * bytes_per_word;
    if (offset > backend.vmem.size() || size > backend.vmem.size() - offset) {
        throw std::runtime_error(
            "VMEM descriptor is outside the bounded in-memory backend");
    }
    return {offset, size};
}

std::pair<std::size_t, std::size_t> checked_dram_range(
    const BoundedMemoryBackend &backend, const LinearDramConfig &config) {
    const std::size_t offset = config.byte_address;
    const std::size_t size = config.byte_count;
    if (offset > backend.dram.size() || size > backend.dram.size() - offset) {
        throw std::runtime_error(
            "DRAM descriptor is outside the bounded in-memory backend");
    }
    return {offset, size};
}

void linear_dram_load(BoundedMemoryBackend *backend,
                      const LinearDramConfig *config,
                      unsigned char *destination) {
    if (!config || (!destination && config->byte_count != 0)) {
        throw std::runtime_error("invalid linear DRAM load arguments");
    }
    const auto [offset, size] = checked_dram_range(*backend, *config);
    std::memcpy(destination, backend->dram.data() + offset, size);
    ++backend->loads;
    std::cerr << "backend.linear_dram_load byte_address=0x" << std::hex
              << config->byte_address << " byte_count=0x"
              << config->byte_count << std::dec << "\n";
}

void linear_dram_store(BoundedMemoryBackend *backend,
                       const LinearDramConfig *config,
                       const unsigned char *source) {
    if (!config || (!source && config->byte_count != 0)) {
        throw std::runtime_error("invalid linear DRAM store arguments");
    }
    const auto [offset, size] = checked_dram_range(*backend, *config);
    std::memcpy(backend->dram.data() + offset, source, size);
    ++backend->stores;
    std::cerr << "backend.linear_dram_store byte_address=0x" << std::hex
              << config->byte_address << " byte_count=0x"
              << config->byte_count << std::dec << "\n";
}

void linear_vmem_load(BoundedMemoryBackend *backend,
                      const LinearVmemConfig *config,
                      std::uint32_t *destination) {
    if (!config || (!destination && config->word_count != 0)) {
        throw std::runtime_error("invalid linear VMEM load arguments");
    }
    const auto [offset, size] = checked_vmem_range(*backend, *config);
    std::memcpy(destination, backend->vmem.data() + offset, size);
    ++backend->loads;
    std::cerr << "backend.linear_vmem_load word_address=0x" << std::hex
              << config->word_address << " word_count=0x"
              << config->word_count << " byte_mask=0x" << config->byte_mask
              << std::dec << "\n";
}

void linear_vmem_store(BoundedMemoryBackend *backend,
                       const LinearVmemConfig *config,
                       const std::uint32_t *source) {
    if (!config || (!source && config->word_count != 0)) {
        throw std::runtime_error("invalid linear VMEM store arguments");
    }
    const auto [offset, size] = checked_vmem_range(*backend, *config);
    const auto *source_bytes = reinterpret_cast<const unsigned char *>(source);
    for (std::size_t index = 0; index < size; ++index) {
        const auto mask = static_cast<unsigned char>(
            config->byte_mask >> ((index % sizeof(std::uint32_t)) * 8));
        backend->vmem[offset + index] = static_cast<unsigned char>(
            (backend->vmem[offset + index] & ~mask) |
            (source_bytes[index] & mask));
    }
    ++backend->stores;
    std::cerr << "backend.linear_vmem_store word_address=0x" << std::hex
              << config->word_address << " word_count=0x"
              << config->word_count << " byte_mask=0x" << config->byte_mask
              << std::dec << "\n";
}

template <typename Function>
void *vtable_entry(Function function) {
    return reinterpret_cast<void *>(function);
}

std::array<void *, 10> make_backend_vtable() {
    return {
        vtable_entry(&backend_destructor),
        vtable_entry(&backend_destructor),
        vtable_entry(&linear_dram_load),
        vtable_entry(&linear_dram_store),
        vtable_entry(&vector_dram_load),
        vtable_entry(&vector_dram_store),
        vtable_entry(&linear_vmem_load),
        vtable_entry(&linear_vmem_store),
        vtable_entry(&vector_vmem_load),
        vtable_entry(&vector_vmem_store),
    };
}

void dump_vtable(const char *label, void *object, std::size_t count) {
    if (!object) {
        std::cerr << label << "=null\n";
        return;
    }
    const auto table = *reinterpret_cast<void ***>(object);
    std::cerr << label << "=0x" << std::hex
              << reinterpret_cast<std::uintptr_t>(object)
              << " vtable=0x" << reinterpret_cast<std::uintptr_t>(table)
              << std::dec << "\n";
    for (std::size_t index = 0; index < count; ++index) {
        Dl_info info{};
        const bool resolved = dladdr(table[index], &info) != 0;
        std::cerr << "  [" << index << "] 0x" << std::hex
                  << reinterpret_cast<std::uintptr_t>(table[index]) << std::dec;
        if (resolved && info.dli_sname) {
            std::cerr << " " << info.dli_sname;
        }
        std::cerr << "\n";
    }
}

std::uintptr_t vp_from_vpdag(void *vpdag) {
    const auto impl = *reinterpret_cast<std::uintptr_t *>(vpdag);
    if (impl == 0) {
        return 0;
    }
    const auto holder = *reinterpret_cast<std::uintptr_t *>(impl + 0x98);
    if (holder == 0) {
        return 0;
    }
    const auto tools = *reinterpret_cast<std::uintptr_t *>(holder + 0x10);
    return tools == 0
        ? 0
        : *reinterpret_cast<std::uintptr_t *>(tools + 0x8);
}

void dump_vp_state(const char *label, std::uintptr_t vp) {
    std::cerr << label << " vp=0x" << std::hex << vp << std::dec;
    if (vp != 0) {
        const auto *bytes = reinterpret_cast<const unsigned char *>(vp);
        std::cerr << " emulator_enabled="
                  << static_cast<unsigned>(bytes[0x138])
                  << " memory_if=0x" << std::hex
                  << *reinterpret_cast<const std::uintptr_t *>(vp + 0x150)
                  << " direct_access=0x"
                  << *reinterpret_cast<const std::uintptr_t *>(vp + 0x160)
                  << " vmem_if=0x"
                  << *reinterpret_cast<const std::uintptr_t *>(vp + 0x260)
                  << " amif=0x"
                  << *reinterpret_cast<const std::uintptr_t *>(vp + 0x288)
                  << " vmem_adapter=0x"
                  << *reinterpret_cast<const std::uintptr_t *>(vp + 0x290)
                  << std::dec;
    }
    std::cerr << "\n";
}

struct PortSpec {
    bool input;
    std::uint32_t split;
    std::string name;
    std::uint32_t byte_offset;
    std::uint32_t size;
    std::uint32_t address;
    bool main_input_output;
};

struct SplitSpec {
    std::uint32_t index;
    std::string name;
    std::uint32_t image_start;
    std::uint32_t image_size;
    std::uint32_t dag_start;
    std::vector<PortSpec> inputs;
    std::vector<PortSpec> outputs;
};

struct ExecutionManifest {
    std::string cavalry_version;
    std::string cavalry_hash;
    std::vector<SplitSpec> splits;
};

std::vector<std::string> split_tabs(const std::string &line) {
    std::vector<std::string> fields;
    std::size_t begin = 0;
    while (true) {
        const auto end = line.find('\t', begin);
        fields.push_back(line.substr(begin, end - begin));
        if (end == std::string::npos) {
            break;
        }
        begin = end + 1;
    }
    return fields;
}

std::uint32_t parse_u32(const std::string &value, const char *label) {
    std::size_t used = 0;
    const auto parsed = std::stoull(value, &used, 10);
    if (used != value.size() || parsed > std::numeric_limits<std::uint32_t>::max()) {
        throw std::runtime_error(std::string("invalid ") + label);
    }
    return static_cast<std::uint32_t>(parsed);
}

ExecutionManifest read_execution_manifest(const std::string &path) {
    std::ifstream stream(path);
    if (!stream) {
        throw std::runtime_error("cannot open " + path);
    }
    ExecutionManifest result;
    std::string line;
    if (!std::getline(stream, line)) {
        throw std::runtime_error("empty ADES manifest");
    }
    const auto header = split_tabs(line);
    if (header.size() != 4 || header[0] != "ADES_MANIFEST" ||
        header[1] != "1") {
        throw std::runtime_error("unsupported ADES manifest header");
    }
    result.cavalry_version = header[2];
    result.cavalry_hash = header[3];
    bool ended = false;
    while (std::getline(stream, line)) {
        if (line == "END") {
            ended = true;
            break;
        }
        const auto fields = split_tabs(line);
        if (fields.empty()) {
            throw std::runtime_error("empty ADES manifest record");
        }
        if (fields[0] == "S") {
            if (fields.size() != 6) {
                throw std::runtime_error("invalid split manifest record");
            }
            const auto index = parse_u32(fields[1], "split index");
            if (index != result.splits.size()) {
                throw std::runtime_error("split indexes are not contiguous");
            }
            const auto image_start = parse_u32(fields[3], "image start");
            const auto image_size = parse_u32(fields[4], "image size");
            const auto dag_start = parse_u32(fields[5], "DAG start");
            if (dag_start < image_start || dag_start >= image_start + image_size) {
                throw std::runtime_error("DAG address is outside split image");
            }
            result.splits.push_back(
                SplitSpec{index, fields[2], image_start, image_size,
                          dag_start, {}, {}});
            continue;
        }
        if (fields[0] != "I" && fields[0] != "O") {
            throw std::runtime_error("unknown ADES manifest record");
        }
        if (fields.size() != 19) {
            throw std::runtime_error("invalid port manifest record");
        }
        const auto split = parse_u32(fields[1], "port split index");
        if (split >= result.splits.size() || result.splits[split].index != split) {
            throw std::runtime_error("port references an unknown split");
        }
        PortSpec port{
            fields[0] == "I",
            split,
            fields[2],
            parse_u32(fields[3], "port byte offset"),
            parse_u32(fields[4], "port size"),
            parse_u32(fields[5], "port address"),
            fields[18] == "1",
        };
        if (port.name.empty() || port.size == 0 ||
            (fields[18] != "0" && fields[18] != "1")) {
            throw std::runtime_error("invalid port manifest values");
        }
        auto &ports = port.input
            ? result.splits[split].inputs
            : result.splits[split].outputs;
        ports.push_back(std::move(port));
    }
    if (!ended || result.splits.empty()) {
        throw std::runtime_error("incomplete ADES manifest");
    }
    std::string trailing;
    if (std::getline(stream, trailing) && !trailing.empty()) {
        throw std::runtime_error("unexpected data after ADES manifest end");
    }
    return result;
}

std::string output_filename(std::uint32_t split, std::size_t output) {
    std::ostringstream name;
    name << "split-";
    if (split < 10) {
        name << '0';
    }
    name << split << "-output-";
    if (output < 10) {
        name << '0';
    }
    name << output << ".bin";
    return name.str();
}

std::string join_path(const std::string &directory, const std::string &name) {
    if (directory.empty()) {
        throw std::runtime_error("empty directory path");
    }
    return directory.back() == '/' ? directory + name : directory + "/" + name;
}

} // namespace

int probe_main(int argc, char **argv) {
    if (argc < 3 || argc > 5) {
        std::cerr << "usage: " << argv[0]
                  << " LIBVASAMIF DAG_BINARY [INPUT_OR_OUTPUT]"
                     " [OUTPUT_PREFIX]\n";
        return 2;
    }

    try {
        void *vas = dlopen(argv[1], RTLD_NOW | RTLD_GLOBAL);
        if (!vas) {
            throw std::runtime_error(dlerror());
        }

        std::cerr << "libraries loaded\n";
        using VpDagCtor = void (*)(void *);
        using VpDagDtor = void (*)(void *);
        using VpDagSetData = void (*)(
            void *, const std::vector<std::uint32_t> &);
        struct VasControl {
            bool enable_coredump;
            bool enable_dag_binary;
        };
        using VpDagSetControl = void (*)(void *, const VasControl &);
        using VpDagParse = int (*)(void *, bool);
        using InitCoreVas = void (*)();
        using VpEmulator = void *(*)(const void *);
        using VpFinishInit = int (*)(void *);
        using AmifLinearLoad = void (*)(void *, unsigned char *,
                                        std::uint32_t, std::uint32_t);
        using AmifLinearStore = void (*)(void *, std::uint32_t,
                                         const unsigned char *, std::uint32_t);
        using AmifRunDag = void (*)(void *, std::uint32_t);

        const auto vpdag_ctor = symbol<VpDagCtor>(vas, "_ZN5VpDagC1Ev");
        const auto vpdag_dtor = symbol<VpDagDtor>(vas, "_ZN5VpDagD1Ev");
        const auto vpdag_set_data = symbol<VpDagSetData>(
            vas, "_ZN5VpDag8dag_dataERKSt6vectorIjSaIjEE");
        const auto vpdag_set_control = symbol<VpDagSetControl>(
            vas, "_ZN5VpDag11vas_controlERK13vas_control_t");
        const auto vpdag_parse = symbol<VpDagParse>(vas, "_ZN5VpDag5parseEb");
        const auto init_core_vas = symbol<InitCoreVas>(
            RTLD_DEFAULT, "_Z13init_core_vasv");
        const auto vp_emulator = symbol<VpEmulator>(
            vas, "_ZNK15VectorProcessor2Vp8emulatorEv");
        const auto vp_finish_init = symbol<VpFinishInit>(
            vas, "_ZN15VectorProcessor2Vp12myFinishInitEv");
        vector_setter = symbol<VectorSetter>(
            vas, "_ZN15VectorProcessor11vp_emu_vmem20vp_emu_vector_setterEPKijN5boost8functionIFvS2_jEEE");
        vector_getter = symbol<VectorGetter>(
            vas, "_ZN15VectorProcessor11vp_emu_vmem20vp_emu_vector_getterEPKiN5boost8functionIFjS2_EEE");

        std::cerr << "symbols resolved\n";
        init_core_vas();
        std::cerr << "core simulator initialized\n";
        const auto words = read_words(argv[2]);
        std::cerr << "DAG words loaded: " << words.size() << "\n";
        constexpr std::size_t production_split0_size = 233212;
        constexpr std::size_t production_split0_dag_offset = 232296;
        constexpr std::size_t production_split1_size = 515564;
        constexpr std::size_t production_split1_dag_offset = 514796;
        const std::size_t source_size = words.size() * sizeof(std::uint32_t);
        const std::vector<std::uint32_t> parser_words =
            source_size == production_split0_size
                ? std::vector<std::uint32_t>(
                      words.begin() +
                          production_split0_dag_offset /
                              sizeof(std::uint32_t),
                      words.end())
                : source_size == production_split1_size
                    ? std::vector<std::uint32_t>(
                          words.begin() +
                              production_split1_dag_offset /
                                  sizeof(std::uint32_t),
                          words.end())
                    : words;
        std::cerr << "parser DAG words: " << parser_words.size() << "\n";

        alignas(16) unsigned char vpdag[64];
        std::memset(vpdag, 0, sizeof(vpdag));
        vpdag_ctor(vpdag);
        std::cerr << "VpDag constructed\n";
        vpdag_set_data(vpdag, parser_words);
        std::cerr << "VpDag data set\n";
        const VasControl control{false, true};
        vpdag_set_control(vpdag, control);
        std::cerr << "VpDag control set (coredump=0, dag_binary=1)\n";
        const auto preparse_vp = vp_from_vpdag(vpdag);
        dump_vp_state("before parse", preparse_vp);
        if (preparse_vp != 0) {
            reinterpret_cast<unsigned char *>(preparse_vp)[0x138] = 1;
            dump_vp_state("before parse after enable", preparse_vp);
        }
        const int result = vpdag_parse(vpdag, false);
        std::cerr << "VpDag parsed\n";

        const auto impl = *reinterpret_cast<std::uintptr_t *>(vpdag);
        const auto holder = *reinterpret_cast<std::uintptr_t *>(impl + 0x98);
        const auto tools = holder == 0
            ? 0
            : *reinterpret_cast<std::uintptr_t *>(holder + 0x10);
        const auto vp = tools == 0
            ? 0
            : *reinterpret_cast<std::uintptr_t *>(tools + 0x8);
        std::cout << "parse_result=" << result << "\n"
                  << "vpdag_impl=0x" << std::hex << impl << "\n"
                  << "tools_holder=0x" << holder << "\n"
                  << "tools_api=0x" << tools << "\n"
                  << "vp=0x" << vp << std::dec << "\n";
        if (vp != 0) {
            auto *vp_bytes = reinterpret_cast<unsigned char *>(vp);
            dump_vp_state("after parse", vp);
            const int finish_result = vp_finish_init(
                reinterpret_cast<void *>(vp));
            std::cerr << "Vp myFinishInit result=" << finish_result << "\n";
            dump_vp_state("after finish init", vp);
            auto backend_vtable = make_backend_vtable();
            BoundedMemoryBackend backend{
                backend_vtable.data(),
                std::vector<unsigned char>(0x200000, 0),
                std::vector<unsigned char>(0x1000000, 0)};
            *reinterpret_cast<void **>(vp + 0x150) = &backend;
            dump_vp_state("after backend install", vp);
            vp_bytes[0x138] = 1;
            void *emulator = vp_emulator(reinterpret_cast<void *>(vp));
            dump_vtable("emulator", emulator, 32);
            const auto amif_linear_load = symbol<AmifLinearLoad>(
                RTLD_DEFAULT, "_ZN5amifa21linear_load_from_vmemEPhjj");
            const auto amif_linear_store = symbol<AmifLinearStore>(
                RTLD_DEFAULT, "_ZN5amifa20linear_store_to_vmemEjPKhj");
            const std::array<unsigned char, 11> witness{
                0x91, 0x82, 0x73, 0x64, 0x55, 0x46,
                0x37, 0x28, 0x19, 0x0a, 0xfb};
            std::array<unsigned char, witness.size()> roundtrip{};
            constexpr std::uint32_t witness_address = 0x203;
            amif_linear_store(emulator, witness_address, witness.data(),
                              static_cast<std::uint32_t>(witness.size()));
            amif_linear_load(emulator, roundtrip.data(), witness_address,
                             static_cast<std::uint32_t>(roundtrip.size()));
            const bool roundtrip_matches = witness == roundtrip;
            std::cout << "backend_roundtrip="
                      << (roundtrip_matches ? "pass" : "fail") << "\n"
                      << "backend_loads=" << backend.loads << "\n"
                      << "backend_stores=" << backend.stores << "\n";
            if (!roundtrip_matches) {
                throw std::runtime_error(
                    "bounded VMEM backend store/load round-trip failed");
            }

            constexpr std::size_t tiny_dag_size = 164;
            const std::size_t dag_size = words.size() * sizeof(std::uint32_t);
            std::vector<unsigned char> executable_dag(dag_size);
            std::memcpy(executable_dag.data(), words.data(), dag_size);
            const auto amif_run_dag = symbol<AmifRunDag>(
                RTLD_DEFAULT, "_ZN5amifa7run_dagEj");

            if (dag_size == tiny_dag_size) {
                constexpr std::size_t input_dbase_offset = 44;
                constexpr std::size_t output_dbase_offset = 104;
                constexpr std::uint32_t input_dram_address = 0x10000;
                constexpr std::uint32_t output_dram_address = 0x10100;
                constexpr std::uint32_t dag_word_address = 0x70000;
                const std::size_t dag_byte_address =
                    static_cast<std::size_t>(dag_word_address) *
                    sizeof(std::uint32_t);
                std::memcpy(executable_dag.data() + input_dbase_offset,
                            &input_dram_address, sizeof(input_dram_address));
                std::memcpy(executable_dag.data() + output_dbase_offset,
                            &output_dram_address, sizeof(output_dram_address));
                std::copy(executable_dag.begin(), executable_dag.end(),
                          backend.vmem.begin() + dag_byte_address);
                std::fill_n(
                    backend.dram.begin() + input_dram_address, 16, 0);
                std::fill_n(
                    backend.dram.begin() + output_dram_address, 16, 0xa5);
                std::cerr << "running tiny DAG at word address 0x" << std::hex
                          << dag_word_address << std::dec << "\n";
                amif_run_dag(emulator, dag_word_address);
                const bool known_answer_matches = std::all_of(
                    backend.dram.begin() + output_dram_address,
                    backend.dram.begin() + output_dram_address + 16,
                    [](unsigned char value) { return value == 1; });
                std::cout << "tiny_known_answer="
                          << (known_answer_matches ? "pass" : "fail") << "\n";
                if (!known_answer_matches) {
                    throw std::runtime_error(
                        "tiny DAG did not produce sixteen bytes of 0x01");
                }
            } else if (dag_size == 233212) {
                constexpr std::uint32_t image_byte_address = 1780376;
                constexpr std::uint32_t dag_byte_address = 2012672;
                constexpr std::uint32_t input_dram_address = 0x100000;
                constexpr std::uint32_t output_dram_address = 0x400000;
                constexpr std::size_t input_dbase_offset = 508;
                constexpr std::size_t output_dbase_offset = 828;
                constexpr std::size_t input_size = 1984512;
                constexpr std::size_t output_size = 1389888;
                const std::size_t dag_offset =
                    dag_byte_address - image_byte_address;
                std::memcpy(
                    executable_dag.data() + dag_offset + input_dbase_offset,
                    &input_dram_address, sizeof(input_dram_address));
                std::memcpy(
                    executable_dag.data() + dag_offset + output_dbase_offset,
                    &output_dram_address, sizeof(output_dram_address));
                std::copy(executable_dag.begin(), executable_dag.end(),
                          backend.vmem.begin() + image_byte_address);
                std::fill_n(backend.dram.begin() + input_dram_address,
                            input_size, 0);
                std::fill_n(backend.dram.begin() + output_dram_address,
                            output_size, 0xa5);
                std::cerr << "running production split 0 at word address 0x"
                          << std::hex
                          << dag_byte_address / sizeof(std::uint32_t)
                          << std::dec << "\n";
                amif_run_dag(
                    emulator,
                    dag_byte_address / sizeof(std::uint32_t));
                const auto unchanged = std::count(
                    backend.dram.begin() + output_dram_address,
                    backend.dram.begin() + output_dram_address + output_size,
                    static_cast<unsigned char>(0xa5));
                const auto output_begin =
                    backend.dram.begin() + output_dram_address;
                const auto output_end = output_begin + output_size;
                const auto nonzero = std::count_if(
                    output_begin, output_end,
                    [](unsigned char value) { return value != 0; });
                std::array<std::size_t, 256> histogram{};
                for (auto cursor = output_begin; cursor != output_end;
                     ++cursor) {
                    ++histogram[*cursor];
                }
                const auto distinct = std::count_if(
                    histogram.begin(), histogram.end(),
                    [](std::size_t count) { return count != 0; });
                std::cout << "production_split=0\n"
                          << "production_output_bytes=" << output_size << "\n"
                          << "production_output_changed_bytes="
                          << output_size - unchanged << "\n"
                          << "production_output_nonzero_bytes=" << nonzero
                          << "\n"
                          << "production_output_distinct_byte_values="
                          << distinct << "\n";
                if (unchanged == output_size) {
                    throw std::runtime_error(
                        "production split 0 did not write its output");
                }
                if (argc == 4) {
                    write_bytes(argv[3], &*output_begin, output_size);
                    std::cout << "production_output_path=" << argv[3]
                              << "\n";
                }
            } else if (dag_size == 515564) {
                if (argc != 5) {
                    throw std::runtime_error(
                        "production split 1 needs INPUT_BINARY and "
                        "OUTPUT_PREFIX");
                }
                constexpr std::uint32_t image_byte_address = 1470704;
                constexpr std::uint32_t dag_byte_address = 1985500;
                constexpr std::uint32_t input_dram_address = 0x100000;
                constexpr std::uint32_t output0_dram_address = 0x400000;
                constexpr std::uint32_t output1_dram_address = 0x600000;
                constexpr std::size_t input_dbase_offset = 28;
                constexpr std::size_t output0_dbase_offset = 572;
                constexpr std::size_t output1_dbase_offset = 680;
                constexpr std::size_t input_size = 1389888;
                constexpr std::size_t output0_size = 1400832;
                constexpr std::size_t output1_size = 919296;
                const std::size_t dag_offset =
                    dag_byte_address - image_byte_address;
                std::memcpy(
                    executable_dag.data() + dag_offset + input_dbase_offset,
                    &input_dram_address, sizeof(input_dram_address));
                std::memcpy(
                    executable_dag.data() + dag_offset + output0_dbase_offset,
                    &output0_dram_address, sizeof(output0_dram_address));
                std::memcpy(
                    executable_dag.data() + dag_offset + output1_dbase_offset,
                    &output1_dram_address, sizeof(output1_dram_address));
                std::copy(executable_dag.begin(), executable_dag.end(),
                          backend.vmem.begin() + image_byte_address);
                const auto input = read_bytes(argv[3]);
                if (input.size() != input_size) {
                    throw std::runtime_error(
                        "production split 1 input size mismatch");
                }
                std::copy(input.begin(), input.end(),
                          backend.dram.begin() + input_dram_address);
                std::fill_n(backend.dram.begin() + output0_dram_address,
                            output0_size, 0xa5);
                std::fill_n(backend.dram.begin() + output1_dram_address,
                            output1_size, 0xa5);
                std::cerr << "running production split 1 at word address 0x"
                          << std::hex
                          << dag_byte_address / sizeof(std::uint32_t)
                          << std::dec << "\n";
                amif_run_dag(
                    emulator,
                    dag_byte_address / sizeof(std::uint32_t));
                const auto output0_begin =
                    backend.dram.begin() + output0_dram_address;
                const auto output1_begin =
                    backend.dram.begin() + output1_dram_address;
                write_bytes(std::string(argv[4]) + ".0.bin",
                            &*output0_begin, output0_size);
                write_bytes(std::string(argv[4]) + ".1.bin",
                            &*output1_begin, output1_size);
                std::cout << "production_split=1\n"
                          << "production_output_0_bytes=" << output0_size
                          << "\nproduction_output_1_bytes=" << output1_size
                          << "\n";
            } else {
                throw std::runtime_error(
                    "unsupported DAG fixture size for this probe");
            }
        }
        vpdag_dtor(vpdag);
        return tools == 0 ? 3 : 0;
    } catch (const std::exception &error) {
        std::cerr << "probe error: " << error.what() << "\n";
        return 1;
    }
}

int main(int argc, char **argv) {
    if (argc != 6) {
        std::cerr << "usage: " << argv[0]
                  << " LIBVASAMIF MANIFEST DVI_DIRECTORY INPUT_TENSOR"
                     " OUTPUT_DIRECTORY\n";
        return 2;
    }

    try {
        const auto manifest = read_execution_manifest(argv[2]);
        const std::string dvi_directory(argv[3]);
        const std::string output_directory(argv[5]);
        if (mkdir(output_directory.c_str(), 0755) != 0 && errno != EEXIST) {
            throw std::runtime_error("cannot create output directory");
        }

        const PortSpec *model_input = nullptr;
        std::size_t dram_size = 0;
        for (const auto &split : manifest.splits) {
            for (const auto &port : split.inputs) {
                if (port.name == "images") {
                    if (model_input != nullptr) {
                        throw std::runtime_error("duplicate model input port");
                    }
                    model_input = &port;
                }
                const auto end = static_cast<std::size_t>(port.address) + port.size;
                dram_size = std::max(dram_size, end);
            }
            for (const auto &port : split.outputs) {
                const auto end = static_cast<std::size_t>(port.address) + port.size;
                dram_size = std::max(dram_size, end);
            }
        }
        if (model_input == nullptr) {
            throw std::runtime_error("missing images model input port");
        }
        if (dram_size > 0x40000000ULL) {
            throw std::runtime_error("manifest DRAM plan exceeds safety bound");
        }
        dram_size = (dram_size + 0xfffU) & ~std::size_t{0xfffU};

        const auto input = read_bytes(argv[4]);
        if (input.size() != model_input->size) {
            throw std::runtime_error(
                "input tensor size does not match manifest images port");
        }

        void *vas = dlopen(argv[1], RTLD_NOW | RTLD_GLOBAL);
        if (!vas) {
            throw std::runtime_error(dlerror());
        }

        using VpDagCtor = void (*)(void *);
        using VpDagDtor = void (*)(void *);
        using VpDagSetData = void (*)(
            void *, const std::vector<std::uint32_t> &);
        struct VasControl {
            bool enable_coredump;
            bool enable_dag_binary;
        };
        using VpDagSetControl = void (*)(void *, const VasControl &);
        using VpDagParse = int (*)(void *, bool);
        using InitCoreVas = void (*)();
        using VpEmulator = void *(*)(const void *);
        using VpFinishInit = int (*)(void *);
        using AmifRunDag = void (*)(void *, std::uint32_t);

        const auto vpdag_ctor = symbol<VpDagCtor>(vas, "_ZN5VpDagC1Ev");
        const auto vpdag_dtor = symbol<VpDagDtor>(vas, "_ZN5VpDagD1Ev");
        const auto vpdag_set_data = symbol<VpDagSetData>(
            vas, "_ZN5VpDag8dag_dataERKSt6vectorIjSaIjEE");
        const auto vpdag_set_control = symbol<VpDagSetControl>(
            vas, "_ZN5VpDag11vas_controlERK13vas_control_t");
        const auto vpdag_parse = symbol<VpDagParse>(
            vas, "_ZN5VpDag5parseEb");
        const auto init_core_vas = symbol<InitCoreVas>(
            RTLD_DEFAULT, "_Z13init_core_vasv");
        const auto vp_emulator = symbol<VpEmulator>(
            vas, "_ZNK15VectorProcessor2Vp8emulatorEv");
        const auto vp_finish_init = symbol<VpFinishInit>(
            vas, "_ZN15VectorProcessor2Vp12myFinishInitEv");
        const auto amif_run_dag = symbol<AmifRunDag>(
            RTLD_DEFAULT, "_ZN5amifa7run_dagEj");
        vector_setter = symbol<VectorSetter>(
            vas,
            "_ZN15VectorProcessor11vp_emu_vmem20vp_emu_vector_setterEPKijN5boost8functionIFvS2_jEEE");
        vector_getter = symbol<VectorGetter>(
            vas,
            "_ZN15VectorProcessor11vp_emu_vmem20vp_emu_vector_getterEPKiN5boost8functionIFjS2_EEE");

        init_core_vas();
        auto backend_vtable = make_backend_vtable();
        BoundedMemoryBackend backend{
            backend_vtable.data(),
            std::vector<unsigned char>(0x400000, 0),
            std::vector<unsigned char>(dram_size, 0)};
        std::copy(input.begin(), input.end(),
                  backend.dram.begin() + model_input->address);
        std::set<std::string> produced{model_input->name};

        std::uint32_t start_split = 0;
        if (const char *value = std::getenv("ADES_START_SPLIT")) {
            start_split = parse_u32(value, "ADES_START_SPLIT");
            if (start_split >= manifest.splits.size()) {
                throw std::runtime_error("ADES_START_SPLIT is out of range");
            }
        }
        if (start_split != 0) {
            const char *resume_directory = std::getenv("ADES_RESUME_DIRECTORY");
            if (resume_directory == nullptr || *resume_directory == '\0') {
                throw std::runtime_error(
                    "ADES_RESUME_DIRECTORY is required when resuming");
            }
            for (const auto &split : manifest.splits) {
                if (split.index >= start_split) {
                    break;
                }
                for (std::size_t index = 0; index < split.outputs.size(); ++index) {
                    const auto &port = split.outputs[index];
                    const auto bytes = read_bytes(join_path(
                        resume_directory,
                        output_filename(split.index, index)));
                    if (bytes.size() != port.size) {
                        throw std::runtime_error(
                            "resume tensor size disagrees with manifest");
                    }
                    std::copy(bytes.begin(), bytes.end(),
                              backend.dram.begin() + port.address);
                    produced.insert(port.name);
                }
            }
        }

        std::cout << "ADES_RESULT\t1\t" << manifest.cavalry_version
                  << "\t" << manifest.cavalry_hash << "\n";
        for (const auto &split : manifest.splits) {
            if (split.index < start_split) {
                continue;
            }
            for (const auto &port : split.inputs) {
                if (produced.find(port.name) == produced.end()) {
                    throw std::runtime_error(
                        "split input has no executed producer: " + port.name);
                }
            }

            const auto dvi_path = join_path(dvi_directory, split.name + ".dvi");
            const auto words = read_words(dvi_path);
            const auto image_size = words.size() * sizeof(std::uint32_t);
            if (image_size != split.image_size) {
                throw std::runtime_error(
                    "DVI image size disagrees with manifest for split " +
                    std::to_string(split.index));
            }
            const std::size_t dag_offset = split.dag_start - split.image_start;
            if (dag_offset % sizeof(std::uint32_t) != 0 ||
                dag_offset >= image_size) {
                throw std::runtime_error("invalid split DAG offset");
            }
            const std::vector<std::uint32_t> parser_words(
                words.begin() + dag_offset / sizeof(std::uint32_t),
                words.end());

            alignas(16) unsigned char vpdag[64]{};
            vpdag_ctor(vpdag);
            vpdag_set_data(vpdag, parser_words);
            const VasControl control{false, true};
            vpdag_set_control(vpdag, control);
            const int parse_result = vpdag_parse(vpdag, false);
            if (parse_result != 0) {
                throw std::runtime_error(
                    "VpDag parse failed for split " +
                    std::to_string(split.index));
            }
            const auto vp = vp_from_vpdag(vpdag);
            if (vp == 0) {
                throw std::runtime_error(
                    "VpDag did not expose a VectorProcessor instance");
            }
            if (vp_finish_init(reinterpret_cast<void *>(vp)) != 0) {
                throw std::runtime_error(
                    "VectorProcessor initialization failed");
            }
            *reinterpret_cast<void **>(vp + 0x150) = &backend;
            reinterpret_cast<unsigned char *>(vp)[0x138] = 1;
            void *emulator = vp_emulator(reinterpret_cast<void *>(vp));
            if (emulator == nullptr) {
                throw std::runtime_error("VectorProcessor emulator is null");
            }

            std::vector<unsigned char> image(image_size);
            std::memcpy(image.data(), words.data(), image_size);
            const auto patch_port = [&](const PortSpec &port) {
                const auto patch = dag_offset + port.byte_offset;
                if (patch > image.size() ||
                    sizeof(port.address) > image.size() - patch) {
                    throw std::runtime_error(
                        "port relocation is outside DVI image");
                }
                std::memcpy(image.data() + patch, &port.address,
                            sizeof(port.address));
            };
            for (const auto &port : split.inputs) {
                patch_port(port);
            }
            for (const auto &port : split.outputs) {
                patch_port(port);
                std::fill_n(backend.dram.begin() + port.address,
                            port.size, 0xa5);
            }
            if (split.image_start > backend.vmem.size() ||
                image.size() > backend.vmem.size() - split.image_start) {
                throw std::runtime_error("DVI image exceeds bounded VMEM");
            }
            std::copy(image.begin(), image.end(),
                      backend.vmem.begin() + split.image_start);

            const auto started = std::chrono::steady_clock::now();
            amif_run_dag(emulator, split.dag_start / sizeof(std::uint32_t));
            const auto elapsed = std::chrono::duration_cast<
                std::chrono::microseconds>(
                std::chrono::steady_clock::now() - started).count();
            std::cout << "SPLIT\t" << split.index << "\t" << elapsed
                      << "\n";

            for (std::size_t index = 0; index < split.outputs.size(); ++index) {
                const auto &port = split.outputs[index];
                const auto begin = backend.dram.begin() + port.address;
                const auto end = begin + port.size;
                const auto unchanged = std::count(
                    begin, end, static_cast<unsigned char>(0xa5));
                if (unchanged == port.size) {
                    throw std::runtime_error(
                        "split did not write output " + port.name);
                }
                const auto path = join_path(
                    output_directory, output_filename(split.index, index));
                write_bytes(path, &*begin, port.size);
                std::cout << "TENSOR\t" << split.index << "\t" << index
                          << "\t" << port.name << "\t" << path
                          << "\t" << port.size << "\t"
                          << port.size - unchanged << "\t"
                          << (port.main_input_output ? 1 : 0) << "\n";
                produced.insert(port.name);
            }
            vpdag_dtor(vpdag);
        }
        std::cout << "END\n";
        return 0;
    } catch (const std::exception &error) {
        std::cerr << "ADES executor error: " << error.what() << "\n";
        return 1;
    }
}
