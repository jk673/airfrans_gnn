#!/usr/bin/env bash

set -euo pipefail

if ! command -v nvidia-smi >/dev/null 2>&1; then
  echo "nvidia-smi not found. NVIDIA driver/CUDA environment is required."
  exit 1
fi

if [[ "${1:-}" == "-h" || "${1:-}" == "--help" ]]; then
  echo "Usage: ./gpu_monitor.sh [--watch seconds]"
  echo "  --watch seconds   Refresh output every N seconds."
  exit 0
fi

WATCH_INTERVAL=""
if [[ "${1:-}" == "--watch" ]]; then
  WATCH_INTERVAL="${2:-2}"
fi

if [[ -t 1 ]]; then
  C_RESET=$'\033[0m'
  C_BOLD=$'\033[1m'
  C_DIM=$'\033[2m'
  C_RED=$'\033[31m'
  C_GREEN=$'\033[32m'
  C_YELLOW=$'\033[33m'
  C_BLUE=$'\033[34m'
  C_MAGENTA=$'\033[35m'
  C_CYAN=$'\033[36m'
else
  C_RESET=""
  C_BOLD=""
  C_DIM=""
  C_RED=""
  C_GREEN=""
  C_YELLOW=""
  C_BLUE=""
  C_MAGENTA=""
  C_CYAN=""
fi

color_by_percent() {
  local pct=$1
  if (( pct < 50 )); then
    printf "%s" "$C_GREEN"
  elif (( pct < 80 )); then
    printf "%s" "$C_YELLOW"
  else
    printf "%s" "$C_RED"
  fi
}

progress_bar() {
  local pct=$1
  local width=24
  local filled=$((pct * width / 100))
  local empty=$((width - filled))
  local color
  color="$(color_by_percent "$pct")"
  printf "%s[" "$color"
  printf "%0.s#" $(seq 1 "$filled")
  printf "%0.s-" $(seq 1 "$empty")
  printf "]%s %3d%%%s" "$C_DIM" "$pct" "$C_RESET"
}

trim() {
  echo "$1" | xargs
}

print_once() {
  clear
  echo -e "${C_BOLD}${C_CYAN}=== NVIDIA GPU Monitor ===${C_RESET}"
  echo -e "${C_DIM}$(date '+%Y-%m-%d %H:%M:%S')${C_RESET}"
  echo

  local query_out
  query_out="$(nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu,power.draw,power.limit --format=csv,noheader,nounits)"

  if [[ -z "$query_out" ]]; then
    echo "No GPU data returned by nvidia-smi."
    return
  fi

  while IFS=',' read -r idx name gpu_util mem_used mem_total temp power_draw power_limit; do
    idx="$(trim "$idx")"
    name="$(trim "$name")"
    gpu_util="$(trim "$gpu_util")"
    mem_used="$(trim "$mem_used")"
    mem_total="$(trim "$mem_total")"
    temp="$(trim "$temp")"
    power_draw="$(trim "$power_draw")"
    power_limit="$(trim "$power_limit")"

    if [[ "$gpu_util" == "N/A" || -z "$gpu_util" ]]; then gpu_util=0; fi
    if [[ "$mem_used" == "N/A" || -z "$mem_used" ]]; then mem_used=0; fi
    if [[ "$mem_total" == "N/A" || -z "$mem_total" ]]; then mem_total=0; fi
    local mem_util
    if (( mem_total > 0 )); then
      mem_util=$((mem_used * 100 / mem_total))
    else
      mem_util=0
    fi

    echo -e "${C_BOLD}${C_MAGENTA}GPU ${idx}${C_RESET} ${C_BOLD}${name}${C_RESET}"
    printf "  %-13s " "Processor:"
    progress_bar "$gpu_util"
    echo
    printf "  %-13s " "vRAM:"
    progress_bar "$mem_util"
    echo
    echo -e "  ${C_BLUE}Memory${C_RESET}: ${mem_used} MiB / ${mem_total} MiB"
    echo -e "  ${C_BLUE}Temp${C_RESET}:   ${temp} C"
    echo -e "  ${C_BLUE}Power${C_RESET}:  ${power_draw} W / ${power_limit} W"
    echo
  done <<< "$query_out"
}

if [[ -n "$WATCH_INTERVAL" ]]; then
  while true; do
    print_once
    sleep "$WATCH_INTERVAL"
  done
else
  print_once
fi
