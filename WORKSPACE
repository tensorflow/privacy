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

dp_lib_release = "2.1.0"
dp_lib_tar_sha256 = "b2e9afb2ea9337bb7c6302545b72e938707e8cdb3558ef38ce5cdd12fe2f182c"
dp_lib_url = "https://github.com/google/differential-privacy/archive/refs/tags/v" + dp_lib_release + ".tar.gz"

http_archive(
    name = "com_google_differential_py",
    sha256 = dp_lib_tar_sha256,
    urls = [
        dp_lib_url,
    ],
    strip_prefix = "differential-privacy-" + dp_lib_release + "/python",
)

# Load transitive dependencies of the DP accounting library.
load("@com_google_differential_py//:accounting_py_deps.bzl", "accounting_py_deps")
accounting_py_deps()
load("@com_google_differential_py//:accounting_py_deps_init.bzl", "accounting_py_deps_init")
accounting_py_deps_init("com_google_differential_py")
