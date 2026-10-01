#pragma once
#include "ggml-backend-impl.h"

// Private device-buffer dispatch. A backend must explicitly support this storage contract.
struct ggml_backend_execution_ops {
    bool (*supports)(void * context, const ggml_tensor * op);
    ggml_status (*compute)(void * context, ggml_backend_t backend, ggml_tensor * op);
    bool (*can_write)(void * context);
    void (*modified)(void * context);
    void (*destroy)(void * context);
    // Bytes this owner needs to execute the op, including internal scratch. Return 0 to decline
    // or when the op is not claimed; the caller then uses stock sizing. Last field on purpose:
    // owners initialize this struct positionally.
    size_t (*alloc_size)(void * context, ggml_backend_buffer_type_t buft, const ggml_tensor * op);
};

// Retain stateless host backing and adopt context only on success; native reset state is not bypassed.
// Destruction callbacks must not throw. Tensor byte layout remains the backing's layout.
GGML_API ggml_backend_buffer_t ggml_backend_execution_buffer_new(ggml_backend_dev_t device,
        ggml_backend_buffer_t backing, const ggml_backend_execution_ops & ops, void * context);
GGML_API bool ggml_backend_buft_is_execution(ggml_backend_buffer_type_t type);
GGML_API bool ggml_backend_execution_buffers_present();
// Multiple unrelated owners are rejected. A null owner means ordinary storage.
GGML_API bool ggml_backend_execution_owner(const ggml_tensor * op, ggml_backend_buffer_t & owner);
GGML_API bool ggml_backend_execution_supports(ggml_backend_buffer_t owner, ggml_backend_dev_t device, const ggml_tensor * op);
GGML_API ggml_status ggml_backend_execution_compute(ggml_backend_buffer_t owner, ggml_backend_t backend, ggml_tensor * op);

// Bytes the owning context needs to execute the op, including internal scratch. Returns 0
// when the owner declines or has no alloc_size callback; the caller then uses stock sizing.
GGML_API size_t ggml_backend_execution_alloc_size(ggml_backend_buffer_t owner,
        ggml_backend_buffer_type_t buft, const ggml_tensor * op);

// Managed attention may omit native output extras only when its graph uses external scratch.
GGML_API void ggml_backend_execution_set_external_workspace(ggml_tensor * op, bool external);
GGML_API bool ggml_backend_execution_has_external_workspace(const ggml_tensor * op);
