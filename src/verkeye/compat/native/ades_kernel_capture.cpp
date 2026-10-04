// Capture ADES's exact, decompressed fast-convolution parameters.
//
// Build inside the digest-pinned ADES image:
//   g++ -std=c++17 -shared -fPIC -O2 -o /work/ades_kernel_capture.so /work/ades_kernel_capture.cpp -ldl
// Run the native executor with:
//   env VERKEYE_FASTCONV_DUMP=/dump LD_PRELOAD=/work/ades_kernel_capture.so /work/ades_executor ...
//
// The 96-byte entries contain process-local vector pointers.  The Python
// normalizer validates these pointer spans and excludes the addresses from the
// portable parameter package.

#include <algorithm>
#include <atomic>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <dlfcn.h>
#include <fstream>
#include <string>
#include <vector>

namespace {

struct Mapping {
  std::uintptr_t start;
  std::uintptr_t end;
  bool readable;
};

std::vector<Mapping> mappings() {
  std::vector<Mapping> result;
  std::ifstream source("/proc/self/maps");
  std::string line;
  while (std::getline(source, line)) {
    unsigned long start = 0;
    unsigned long end = 0;
    char permissions[5] = {};
    if (std::sscanf(line.c_str(), "%lx-%lx %4s", &start, &end, permissions) ==
        3) {
      result.push_back({start, end, permissions[0] == 'r'});
    }
  }
  return result;
}

bool readable_span(const std::vector<Mapping>& ranges, std::uintptr_t start,
                   std::uintptr_t end) {
  if (start == 0 || end < start || end - start > (1U << 28)) {
    return false;
  }
  return std::any_of(ranges.begin(), ranges.end(), [&](const Mapping& mapping) {
    return mapping.readable && start >= mapping.start && end <= mapping.end;
  });
}

void write_bytes(const std::string& path, const void* data, std::size_t size) {
  std::FILE* output = std::fopen(path.c_str(), "wb");
  if (output == nullptr || (size != 0 && std::fwrite(data, 1, size, output) != size) ||
      std::fclose(output) != 0) {
    std::fprintf(stderr, "VERKEYE_FASTCONV cannot write %s\n", path.c_str());
    std::abort();
  }
}

}  // namespace

extern "C" void verkeye_fastconv_exec(void* self)
    asm("_ZN16node_fastconv_op4execEv");

extern "C" void verkeye_fastconv_exec(void* self) {
  using Exec = void (*)(void*);
  static Exec original = reinterpret_cast<Exec>(
      dlsym(RTLD_NEXT, "_ZN16node_fastconv_op4execEv"));
  static std::atomic<unsigned> sequence{0};
  if (original == nullptr) {
    std::fprintf(stderr, "VERKEYE_FASTCONV cannot resolve original exec\n");
    std::abort();
  }
  const char* dump_root = std::getenv("VERKEYE_FASTCONV_DUMP");
  if (dump_root == nullptr || *dump_root == '\0') {
    std::fprintf(stderr, "VERKEYE_FASTCONV_DUMP is not set\n");
    std::abort();
  }

  const unsigned call = sequence.fetch_add(1, std::memory_order_relaxed);
  const auto ranges = mappings();
  const auto* object = static_cast<const std::uint8_t*>(self);
  const auto* header = reinterpret_cast<const std::uintptr_t*>(object + 0x420);
  const std::uintptr_t begin = header[0];
  const std::uintptr_t end = header[1];
  if (!readable_span(ranges, begin, end) || (end - begin) % 96 != 0) {
    std::fprintf(stderr, "VERKEYE_FASTCONV invalid entry span at call %u\n", call);
    std::abort();
  }

  const std::string prefix = std::string(dump_root) + "/fastconv-" +
                             std::to_string(call);
  write_bytes(prefix + "-entries.bin", reinterpret_cast<const void*>(begin),
              end - begin);
  const std::size_t count = (end - begin) / 96;
  for (std::size_t entry_index = 0; entry_index < count; ++entry_index) {
    const auto* entry =
        reinterpret_cast<const std::uint8_t*>(begin + entry_index * 96);
    const auto* vector = reinterpret_cast<const std::uintptr_t*>(entry);
    const std::uintptr_t vector_begin = vector[0];
    const std::uintptr_t vector_end = vector[1];
    const std::uintptr_t vector_capacity = vector[2];
    if (vector_end < vector_begin || vector_capacity < vector_end) {
      std::fprintf(stderr,
                   "VERKEYE_FASTCONV invalid point span call=%u entry=%zu\n",
                   call, entry_index);
      std::abort();
    }
    const std::size_t size = vector_end - vector_begin;
    const std::string path = prefix + "-entry-" +
                             std::to_string(entry_index) + "-offset-0.bin";
    if (size == 0) {
      write_bytes(path, nullptr, 0);
    } else if (size % 24 == 0 && readable_span(ranges, vector_begin, vector_end)) {
      write_bytes(path, reinterpret_cast<const void*>(vector_begin), size);
    } else {
      std::fprintf(stderr,
                   "VERKEYE_FASTCONV unreadable point span call=%u entry=%zu\n",
                   call, entry_index);
      std::abort();
    }
  }
  std::fprintf(stderr, "VERKEYE_FASTCONV call=%u channels=%zu\n", call, count);
  original(self);
}
