package(default_visibility = ["//visibility:public"])

licenses(["notice"])

exports_files(["LICENSE"])

# This is here for backwards compatibility. New BUILD rules should depend on
# //third_party/py/tensorflow_privacy:privacy directly.
py_library(
    name = "privacy",
    deps = [
        "//third_party/py/tensorflow_privacy",
    ],
)
