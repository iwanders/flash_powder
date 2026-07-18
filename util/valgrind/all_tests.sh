#!/bin/bash -xe

DIR=$( cd "$( dirname "${BASH_SOURCE[0]}" )" && pwd )

ALL_TESTS=$(cargo test --no-run --message-format=json | jq -r 'select(.profile.test == true and .executable != null) | .executable')

for line in ${ALL_TESTS}; do
    echo "Processing: $line"
    ${DIR}/valgrind.sh ${line}
done

echo "🎉"
