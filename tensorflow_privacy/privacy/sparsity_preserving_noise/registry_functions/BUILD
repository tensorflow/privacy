package(default_visibility = ["//visibility:public"])

licenses(["notice"])

py_library(
    name = "embedding",
    srcs = ["embedding.py"],
    deps = ["//tensorflow_privacy/privacy/sparsity_preserving_noise:type_aliases"],
)

py_test(
    name = "embedding_test",
    srcs = ["embedding_test.py"],
    deps = [":embedding"],
)
