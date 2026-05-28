#!/bin/bash -xe

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

valgrind -s  --error-exitcode=1  --leak-check=full  --suppressions="${DIR}"/combined.supp  --suppressions="${DIR}"/manual.supp --suppressions="${DIR}"/rust_generic.supp $*
