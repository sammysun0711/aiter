// SPDX-License-Identifier: MIT
// Test-process allocator: mappings are released when the subprocess exits.
#include <hip/hip_runtime_api.h>
#include <cstddef>
#include <cstdint>

static int allocate_guarded_hip_impl(size_t bytes, int device, void** result,
                                     uint64_t low_bits) {
    hipMemAllocationProp prop{};
    prop.type = hipMemAllocationTypePinned;
    prop.location.type = hipMemLocationTypeDevice;
    prop.location.id = device;
    size_t granularity = 0;
    auto status = hipMemGetAllocationGranularity(&granularity, &prop, hipMemAllocationGranularityMinimum);
    if (status != hipSuccess) return status;
    size_t mapped = ((bytes + granularity - 1) / granularity) * granularity;
    void* base = nullptr;
    // Choose the low-32 VA band independently of allocation history.
    const size_t virtual_span = size_t{8} << 30;
    status = hipMemAddressReserve(&base, virtual_span, granularity, nullptr, 0);
    if (status != hipSuccess) return status;
    const auto aligned = (reinterpret_cast<uintptr_t>(base) + (uint64_t{1} << 32) - 1) & ~((uint64_t{1} << 32) - 1);
    void* mapped_base = reinterpret_cast<void*>(aligned + low_bits);
    hipMemGenericAllocationHandle_t handle;
    status = hipMemCreate(&handle, mapped, &prop, 0);
    if (status != hipSuccess) return status;
    status = hipMemMap(mapped_base, mapped, 0, handle, 0);
    if (status != hipSuccess) return status;
    hipMemAccessDesc access{};
    access.location = prop.location;
    access.flags = hipMemAccessFlagsProtReadWrite;
    status = hipMemSetAccess(mapped_base, mapped, &access, 1);
    if (status != hipSuccess) return status;
    *result = static_cast<char*>(mapped_base) + mapped - bytes;
    return hipSuccess;
}

extern "C" int allocate_guarded_hip(size_t bytes, int device, void** result) {
    return allocate_guarded_hip_impl(bytes, device, result, 0x90000000ULL);
}

extern "C" int allocate_guarded_hip_buffer(size_t bytes, int device, void** result) {
    return allocate_guarded_hip_impl(bytes, device, result, 0x10000000ULL);
}
