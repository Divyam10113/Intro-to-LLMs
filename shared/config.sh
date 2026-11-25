# Central configuration for shell scripts. Source it:  . "<repo>/shared/config.sh"
#
# Order of precedence for every setting: the environment, then <repo>/.env (lines KEY=value,
# git-ignored; copy .env.example), then the defaults below. The defaults reproduce the paths used
# on the course server, where this repository sits at proj/Intro-to-LLMs/ next to proj/MVFA-AD/
# and proj/.venv-task4/.
_cfg_dir=$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)
: "${REPO_ROOT:=$(dirname "$_cfg_dir")}"
export REPO_ROOT

if [ -f "$REPO_ROOT/.env" ]; then
  while IFS= read -r _line || [ -n "$_line" ]; do
    case "$_line" in ''|'#'*) continue ;; esac
    _key=${_line%%=*}; _key=${_key#export }; _key=$(echo "$_key" | tr -d '[:space:]')
    _val=${_line#*=}; _val=${_val%\"}; _val=${_val#\"}; _val=${_val%\'}; _val=${_val#\'}
    [ -n "$_key" ] && [ -z "${!_key+x}" ] && export "$_key=$_val"
  done < "$REPO_ROOT/.env"
  unset _line _key _val
fi

: "${PROJ_ROOT:=$(dirname "$REPO_ROOT")}"                      # folder that holds the repo
: "${TASK4_PYTHON:=$PROJ_ROOT/.venv-task4/bin/python}"        # Task 4 environment (torch, pytest)
: "${MVFA_DIR:=$PROJ_ROOT/MVFA-AD}"                           # MVFA-AD checkout, commit a30bf7b
: "${TASK4_HDD:=/mnt/HDD1/divyam/task4}"                      # Task 4 data, checkpoints, runs
: "${TASK4_DL_TMP:=$PROJ_ROOT/.dl_tmp}"                       # scratch for download_data.sh
export PROJ_ROOT TASK4_PYTHON MVFA_DIR TASK4_HDD TASK4_DL_TMP
unset _cfg_dir
