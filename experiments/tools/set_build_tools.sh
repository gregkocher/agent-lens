#!/usr/bin/env bash
# Put a pod (image runpod/base:0.6.2-cpu) into a known build-tools state. Run ON the pod as
# root, AFTER pod_bootstrap.sh / uv sync. The state applies to the whole pod, so never run
# sweeps with different build_tools on one pod. Usage: set_build_tools.sh on|off|check
#   on  : compilers + Python headers (image default) + Cython and pybind11 for system python.
#         Matches the tool set of the original optimize_cache Dockerfile.
#   off : python + pip only. Compilers, binutils, make/cmake and Python headers are purged.
# PyPI stays reachable in both states (prebuilt C wheels such as lru-dict are still possible).
set -uo pipefail
export DEBIAN_FRONTEND=noninteractive
mode=${1:?usage: set_build_tools.sh on|off|check}
sys_py=/usr/bin/python3

check() {
  # Prints one line per tool; exit status says whether the pod matches the requested state.
  local want=$1 bad=0
  for t in gcc g++ cc c++ cpp as ld make cmake; do
    if command -v "$t" >/dev/null 2>&1; then have=1; else have=0; fi
    echo "tool $t present=$have"
    [ "$want" = off ] && [ $have = 1 ] && bad=1
    [ "$want" = on ] && [ $have = 0 ] && bad=1
  done
  for m in Cython pybind11; do
    $sys_py -c "import $m" >/dev/null 2>&1 && have=1 || have=0
    echo "module $m present=$have"
    [ "$want" = off ] && [ $have = 1 ] && bad=1
    [ "$want" = on ] && [ $have = 0 ] && bad=1
  done
  hdr=$($sys_py -c "import sysconfig,os;print(int(os.path.exists(os.path.join(sysconfig.get_paths()['include'],'Python.h'))))")
  echo "header Python.h present=$hdr"
  [ "$want" = off ] && [ "$hdr" = 1 ] && bad=1
  [ "$want" = on ] && [ "$hdr" = 0 ] && bad=1
  return $bad
}

case "$mode" in
  on)
    $sys_py -m pip install --quiet --disable-pip-version-check "cython==3.2.9" "pybind11==2.13.6"
    check on
    ;;
  off)
    # Exact installed package names through dpkg: apt reads 'gcc-*' as a regex matching most
    # of the system, and -dev library packages in the image would pull removals along.
    pkgs=$(dpkg-query -W -f='${Package}\n' | grep -E '^(build-essential|gcc(-[0-9]+)?|g\+\+(-[0-9]+)?|cpp(-[0-9]+)?|gfortran(-[0-9]+)?|binutils|binutils-x86-64-linux-gnu|make|cmake|dpkg-dev|(lib)?python3(\.[0-9]+)?-dev)$')
    [ -z "$pkgs" ] || dpkg --purge --force-depends $pkgs >/dev/null 2>&1
    # The image also ships Homebrew GCC outside PATH (agents have found and run it).
    rm -rf /home/linuxbrew
    # Compilers not owned by a package in the image would survive the purge, so check() decides.
    check off
    ;;
  check)
    check "${2:?usage: set_build_tools.sh check on|off}"
    ;;
  *) echo "unknown mode $mode" >&2; exit 2 ;;
esac
rc=$?
echo "BUILD_TOOLS_STATE mode=$mode rc=$rc"
exit $rc
