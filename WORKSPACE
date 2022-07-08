workspace(name = "org_tensorflow_privacy")

load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

git_repository(
    name = "bazel_skylib",
    remote = "https://github.com/bazelbuild/bazel-skylib.git",
    tag = "1.0.3",
)

git_repository(
    name = "rules_python",
    remote = "https://github.com/bazelbuild/rules_python.git",
    tag = "0.5.0",
)

dp_lib_commit = "3485dec02794bda52d7b0ea248fa2a2caf11d55b"
dp_lib_tar_sha256 = "e2dfa8a42b7f6a4b701a9a0ea0904e0b716e2ace691dcd789af2e9185f21f4b8"
dp_lib_url = "https://github.com/google/differential-privacy/archive/" + dp_lib_commit + ".tar.gz"

http_archive(
    name = "com_google_differential_py",
    sha256 = dp_lib_tar_sha256,
    urls = [
        dp_lib_url,
    ],
    strip_prefix = "differential-privacy-" + dp_lib_commit + "/python",
)

# Load transitive dependencies of the DP accounting library.
load("@com_google_differential_py//:accounting_py_deps.bzl", "accounting_py_deps")
accounting_py_deps()
load("@com_google_differential_py//:accounting_py_deps_init.bzl", "accounting_py_deps_init")
accounting_py_deps_init("com_google_differential_py")
