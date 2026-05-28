#!/bin/bash -xe

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

# Gen suppressions;
#valgrind -s --gen-suppressions=all --leak-check=full  --suppressions="${DIR}"/combined.supp  --suppressions="${DIR}"/manual.supp --suppressions="${DIR}"/rust_generic.supp $*
# Run the test:
valgrind -s  --error-exitcode=1  --leak-check=full  --suppressions="${DIR}"/combined.supp  --suppressions="${DIR}"/manual.supp --suppressions="${DIR}"/rust_generic.supp $*
