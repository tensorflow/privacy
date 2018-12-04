licenses(["notice"])  # Apache 2.0

py_library(
    name = "gaussian_average_query",
    srcs = ["gaussian_average_query.py"],
    deps = [
        ":private_queries",
        "//third_party/py/tensorflow",
    ],
)

py_library(
    name = "dp_optimizers",
    deps = [
        ":dp_adam",
        ":dp_gradient_descent",
    ],
)

py_library(
    name = "dp_adam",
    srcs = [
        "dp_adam.py",
    ],
    deps = [
        ":gaussian_average_query",
        "//third_party/py/tensorflow",
    ],
)

py_library(
    name = "dp_gradient_descent",
    srcs = [
        "dp_gradient_descent.py",
    ],
    deps = [
        ":gaussian_average_query",
        "//third_party/py/tensorflow",
    ],
)

py_test(
    name = "dp_optimizer_test",
    srcs = ["dp_optimizer_test.py"],
    deps = [
        ":dp_optimizers",
        "//third_party/py/absl/testing:parameterized",
        "//third_party/py/tensorflow",
    ],
)

py_library(
    name = "private_queries",
    srcs = ["private_queries.py"],
)
