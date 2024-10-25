#!/bin/bash
# SPDX-License-Identifier: Apache-2.0

set -eo pipefail

# This script exists to upload data to s3 and render the final markdown
# file for the results of the benchmarking.

function upload_to_s3() {
    local -r bucket_name="$1"
    local -r file_path="$2"
    local -r destination_path="$3"

    local -r bucket_path="s3://${bucket_name}/${destination_path}"
    printf 'Uploading result to S3: %s\n' "${bucket_path}"
    if [[ ! -f "${file_path}" ]]; then
        echo "Error: File '${file_path}' does not exist."
        exit 1
    fi
    aws s3 cp "${file_path}" "${bucket_path}"
}

################################################################################
# Returns the path to where we'll be uploading the loss.png file to.
# Currently, the format is in the form of:
# pulls/<base_branch>/<pr_number>/<sha>/loss.png
# This way, a single PR can have multiple runs and we can keep track of them.
# Globals:
#   github (read-only) - The github context
# Arguments:
#  None
# Returns:
#  (string) The path to where we'll be uploading the loss.png file to.
################################################################################
function get_s3_path() {
    printf 'pulls/%s/%s/%s/loss.png' "${{ github.event.pull_request.base.ref }}" "${{ github.event.pull_request.number }}" "${{ github.event.pull_request.head.sha}}"
}

function export_results() {
    local -r img_url="$1"
    printf '### Test performance:\n\n![Loss curve](%s)\n' "${img_url}" >> "${GITHUB_STEP_SUMMARY}"
}

function main() {
    local -r output_path=$(get_s3_path)
    local -r bucket_name='os-ci-loss-curve-test'
    local -r access_region="${{ vars.AWS_REGION }}"
    local -r input_file='./loss.png'
    local -r final_url="https://${bucket_name}.s3.${access_region}.amazonaws.com/${output_path}"

    printf 'Uploading image "%s" to bucket "%s" at output path "%s"\n' "${input_file}" "${bucket_name}" "${output_path}"
    upload_to_s3 "${bucket_name}" "${input_file}" "${output_path}"

    printf 'Final url should be: "%s"\n' "${final_url}"
    export_results "${final_url}"
}

main
