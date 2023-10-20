#!/bin/bash
# set -ex

day_config_diff=0

function calc_day() {
  system=$(uname)
  if [ ${system} == "Linux" ]; then
    current_day_mdy=$(TZ=UTC-8 date -d "-${day_config_diff} days" +%m-%d-%Y)
    current_day_ymd=$(TZ=UTC-8 date -d "-${day_config_diff} days" +%Y-%m-%d)
    current_ts=$(TZ=UTC-8 date -d "${current_day_ymd} 00:00:00" +%s)
  elif [ ${system} == "Darwin" ]; then
    current_day_mdy=$(TZ=UTC-8 date -v-${day_config_diff}d +%m-%d-%Y)
    current_day_ymd=$(TZ=UTC-8 date -v-${day_config_diff}d +%Y-%m-%d)
    current_ts=$(TZ=UTC-8 date -j -f "%Y-%m-%d %H:%M:%S" "${current_day_ymd} 00:00:00" +%s)
  else
    echo "Unknow System."
    exit 1
  fi
  echo "current_day_mdy:${current_day_mdy}"
}

function switch_commit() {
  cur_path=`pwd`
  [[ -z $1 ]] && echo "Please input Paddle absolute path" && exit 1
  CODE_DIR=$1
  [[ -n $2 ]] && day_config_diff=$2
  DAY_SECONDS=86400
  cd ${CODE_DIR} || exit 1
  calc_day
  last_commit_day=$(TZ=UTC-8 git log --before="${current_day_mdy} 00:00:00" -1 --pretty=format:"%cr")
  last_commit_ts=$(TZ=UTC-8 git log --before="${current_day_mdy} 00:00:00" -1 --pretty=format:"%at")
  echo "last_commit_date: ${last_commit_day}"
  echo "last_commit_timestamp: ${last_commit_ts}"
  time_diff=$((current_ts - last_commit_ts))
  if [ $time_diff -gt $DAY_SECONDS ]; then
    echo "Time passed over a day, this commit had been compiled."
    exit 0
  fi
  COMMIT=$(TZ=UTC-8 git log --before="${current_day_mdy} 00:00:00" -1 --pretty=format:"%H")
#   git checkout ${COMMIT}
  echo paddle_commit is ${COMMIT}
  cd ${cur_path}
  echo ${COMMIT} > paddle_commit.log
}
switch_commit $1 $2
