load("@rules_python//python:defs.bzl", "py_library")

package(default_visibility = ["//visibility:public"])

py_library(
    name = "gradient_clipping_utils",
    srcs = ["gradient_clipping_utils.py"],
    srcs_version = "PY3",
)

py_library(
    name = "layer_registry_factories",
    srcs = ["layer_registry_factories.py"],
    srcs_version = "PY3",
)

py_library(
    name = "clip_grads",
    srcs = ["clip_grads.py"],
    srcs_version = "PY3",
    deps = [
        ":gradient_clipping_utils",
        ":layer_registry_factories",
    ],
)
