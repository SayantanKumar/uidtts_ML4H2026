#!/bin/bash

# Shared helpers for running either all cases or a hard-coded subset.
# Each batch script should define:
#   FILE_SELECTION_MODE="all"  # all|subset
#   SELECTED_INPUT_FILES=( "11121324.txt" )

selection_case_base() {
    local name
    name="$(basename "$1")"
    name="${name%.gz}"
    name="${name%.csv}"
    name="${name%.txt}"
    name="${name%.bsv}"
    name="${name%.json}"
    name="${name%_timeline_extracted}"
    name="${name%_results}"
    name="${name%_summary}"
    name="${name%_positions_1}"
    name="${name%_positions}"
    name="${name%_timelines}"
    name="${name%_timeline}"
    name="${name%_extracted}"
    name="${name%_joined}"
    name="${name%_matches}"
    name="${name%_tags}"
    name="${name%_modified}"
    echo "$name"
}

selection_numeric_tokens() {
    printf "%s\n" "$1" | grep -Eo '[0-9]{5,}' || true
}

selection_match_candidates() {
    local base="$1"
    local token

    echo "$base"
    while IFS= read -r token; do
        [[ -n "$token" ]] && echo "$token"
    done < <(selection_numeric_tokens "$base")
}

selection_bases_match() {
    local observed="$1"
    local selected="$2"
    local token

    [[ "$observed" == "$selected" ]] && return 0

    while IFS= read -r token; do
        [[ -n "$token" ]] || continue
        [[ "$selected" == "$token" ]] && return 0
    done < <(selection_numeric_tokens "$observed")

    while IFS= read -r token; do
        [[ -n "$token" ]] || continue
        [[ "$observed" == "$token" ]] && return 0
    done < <(selection_numeric_tokens "$selected")

    return 1
}

init_file_selection() {
    FILE_SELECTION_MODE="${FILE_SELECTION_MODE:-all}"
    SELECTED_CASE_BASES=()

    case "$FILE_SELECTION_MODE" in
        all)
            return 0
            ;;
        subset)
            local declaration
            declaration="$(declare -p SELECTED_INPUT_FILES 2>/dev/null || true)"
            [[ "$declaration" == declare\ -a* ]] || die "SELECTED_INPUT_FILES must be a Bash array when FILE_SELECTION_MODE=subset"
            [[ "${#SELECTED_INPUT_FILES[@]}" -gt 0 ]] || die "SELECTED_INPUT_FILES is empty while FILE_SELECTION_MODE=subset"

            local item
            for item in "${SELECTED_INPUT_FILES[@]}"; do
                SELECTED_CASE_BASES+=("$(selection_case_base "$item")")
            done
            ;;
        *)
            die "FILE_SELECTION_MODE must be either 'all' or 'subset'; got '$FILE_SELECTION_MODE'"
            ;;
    esac
}

selection_enabled() {
    [[ "${FILE_SELECTION_MODE:-all}" == "subset" ]]
}

is_selected_base() {
    local base="$1"

    if ! selection_enabled; then
        return 0
    fi

    local selected
    for selected in "${SELECTED_CASE_BASES[@]}"; do
        selection_bases_match "$base" "$selected" && return 0
    done

    return 1
}

filter_selected_files_in_place() {
    local array_name="$1"

    if ! selection_enabled; then
        return 0
    fi

    local files_copy=()
    eval "files_copy=(\"\${${array_name}[@]}\")"

    local kept=()
    local file
    local base
    for file in "${files_copy[@]}"; do
        base="$(selection_case_base "$file")"
        if is_selected_base "$base"; then
            kept+=("$file")
        fi
    done

    eval "$array_name=()"
    for file in "${kept[@]}"; do
        eval "$array_name+=(\"\$file\")"
    done
}

prepare_effective_input_dir() {
    EFFECTIVE_INPUT_DIR="$INPUT_DIR"

    if ! selection_enabled; then
        return 0
    fi

    local work_root="${SELECTION_WORK_DIR:-$OUTPUT_DIR/selection_work}"
    EFFECTIVE_INPUT_DIR="$work_root/input_${SLURM_JOB_ID:-manual}_$$"
    mkdir -p "$EFFECTIVE_INPUT_DIR"

    local item
    local src
    local base
    local linked=0
    for item in "${SELECTED_INPUT_FILES[@]}"; do
        src="$INPUT_DIR/$item"
        if [[ ! -e "$src" && "$item" != *.txt ]]; then
            src="$INPUT_DIR/${item}.txt"
        fi
        if [[ ! -e "$src" && "$item" != *.txt.gz ]]; then
            src="$INPUT_DIR/${item}.txt.gz"
        fi
        [[ -e "$src" ]] || die "Selected input file not found under INPUT_DIR: $item"

        base="$(selection_case_base "$src")"
        is_selected_base "$base" || die "Internal selection error for selected input: $item"

        ln -sfn "$src" "$EFFECTIVE_INPUT_DIR/$(basename "$src")"
        linked=$((linked + 1))
    done

    [[ "$linked" -gt 0 ]] || die "No selected input files were staged"
}

stage_selected_files_for_bases() {
    local dest_dir="$1"
    shift

    STAGED_SELECTED_DIR="$dest_dir"
    mkdir -p "$STAGED_SELECTED_DIR"

    if ! selection_enabled; then
        return 0
    fi

    local base
    local candidate
    local template
    local src
    local prefix
    local suffix
    local matches
    local found
    for base in "${SELECTED_CASE_BASES[@]}"; do
        found=0
        for template in "$@"; do
            src="${template//\{\}/$base}"
            if [[ -e "$src" ]]; then
                ln -sfn "$src" "$STAGED_SELECTED_DIR/$(basename "$src")"
                found=1
                break
            fi

            prefix="${template%%\{\}*}"
            suffix="${template#*\{\}}"
            while IFS= read -r candidate; do
                [[ -n "$candidate" ]] || continue
                matches=("${prefix}"*"${candidate}"*"${suffix}")
                for src in "${matches[@]}"; do
                    [[ -e "$src" ]] || continue
                    if is_selected_base "$(selection_case_base "$src")"; then
                        ln -sfn "$src" "$STAGED_SELECTED_DIR/$(basename "$src")"
                        found=1
                        break
                    fi
                done
                [[ "$found" -eq 1 ]] && break
            done < <(selection_match_candidates "$base")

            [[ "$found" -eq 1 ]] && break
        done

        [[ "$found" -eq 1 ]] || die "Could not find selected case '$base' using expected file patterns"
    done
}

log_file_selection() {
    log "File selection mode: ${FILE_SELECTION_MODE:-all}"

    if selection_enabled; then
        log "Selected case bases: ${SELECTED_CASE_BASES[*]}"
        if [[ -n "${EFFECTIVE_INPUT_DIR:-}" ]]; then
            log "Effective input directory: $EFFECTIVE_INPUT_DIR"
        fi
    fi
}