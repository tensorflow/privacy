package(default_visibility = ["//visibility:public"])

py_library(
    name = "type_aliases",
    srcs = ["type_aliases.py"],
    srcs_version = "PY3",
)

py_library(
    name = "common_manip_utils",
    srcs = ["common_manip_utils.py"],
    srcs_version = "PY3",
    deps = [":type_aliases"],
)

py_library(
    name = "common_test_utils",
    srcs = ["common_test_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":clip_grads",
        ":layer_registry",
        ":type_aliases",
    ],
)

py_library(
    name = "gradient_clipping_utils",
    srcs = ["gradient_clipping_utils.py"],
    srcs_version = "PY3",
    deps = [
        ":layer_registry",
        ":type_aliases",
    ],
)

py_test(
    name = "gradient_clipping_utils_test",
    srcs = ["gradient_clipping_utils_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":gradient_clipping_utils",
        ":type_aliases",
    ],
)

py_library(
    name = "layer_registry",
    srcs = ["layer_registry.py"],
    srcs_version = "PY3",
    deps = [
        ":type_aliases",
        "//tensorflow_privacy/privacy/fast_gradient_clipping/registry_functions:dense",
        "//tensorflow_privacy/privacy/fast_gradient_clipping/registry_functions:embedding",
    ],
)

py_library(
    name = "clip_grads",
    srcs = ["clip_grads.py"],
    srcs_version = "PY3",
    deps = [
        ":common_manip_utils",
        ":gradient_clipping_utils",
        ":layer_registry",
        ":type_aliases",
    ],
)

py_test(
    name = "clip_grads_test",
    srcs = ["clip_grads_test.py"],
    python_version = "PY3",
    srcs_version = "PY3",
    deps = [
        ":clip_grads",
        ":common_test_utils",
        ":layer_registry",
        ":type_aliases",
    ],
)
