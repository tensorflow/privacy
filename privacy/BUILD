package(default_visibility = ["//visibility:public"])

licenses(["notice"])  # Apache 2.0

exports_files(["LICENSE"])

py_library(
    name = "privacy",
    srcs = ["__init__.py"],
    deps = [
        "//third_party/py/tensorflow_privacy/privacy/analysis:privacy_ledger",
        "//third_party/py/tensorflow_privacy/privacy/dp_query",
        "//third_party/py/tensorflow_privacy/privacy/dp_query:gaussian_query",
        "//third_party/py/tensorflow_privacy/privacy/dp_query:nested_query",
        "//third_party/py/tensorflow_privacy/privacy/dp_query:no_privacy_query",
        "//third_party/py/tensorflow_privacy/privacy/dp_query:normalized_query",
        "//third_party/py/tensorflow_privacy/privacy/dp_query:quantile_adaptive_clip_sum_query",
        "//third_party/py/tensorflow_privacy/privacy/optimizers:dp_optimizer",
    ],
)
