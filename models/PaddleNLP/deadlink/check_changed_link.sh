#!/bin/sh

REPO=$1
BRANCH=$2

python3 test_deadlink.py \
  --code_path ${REPO} \
  --repo ${REPO} \
  --branch ${BRANCH} \
  --func get_link

mv ${REPO} ${REPO}_pr
mv ${REPO}_base ${REPO}
mv ${REPO}_${BRANCH}_md_midoutput.xlsx ${REPO}_pr_${BRANCH}_md_midoutput.xlsx
mv ${REPO}_${BRANCH}_md_midoutput.txt ${REPO}_pr_${BRANCH}_md_midoutput.txt
mv ${REPO}_${BRANCH}_md_midoutput.pkl ${REPO}_pr_${BRANCH}_md_midoutput.pkl

#get link in base code
python3 test_deadlink.py \
  --code_path ${REPO} \
  --repo ${REPO} \
  --branch ${BRANCH} \
  --func get_link

mv ${REPO}_${BRANCH}_md_midoutput.xlsx ${REPO}_base_${BRANCH}_md_midoutput.xlsx
mv ${REPO}_${BRANCH}_md_midoutput.txt ${REPO}_base_${BRANCH}_md_midoutput.txt
mv ${REPO}_${BRANCH}_md_midoutput.pkl ${REPO}_base_${BRANCH}_md_midoutput.pkl

#get chaged link by pr
python3 test_deadlink.py \
  --link_new_file ${REPO}_pr_${BRANCH}_md_midoutput.xlsx \
  --link_old_file ${REPO}_base_${BRANCH}_md_midoutput.xlsx \
  --link_diff_file ${REPO}_${BRANCH}_md_diff.xlsx \
  --func diff_link

#check link
python3 test_deadlink.py \
  --link_file ${REPO}_${BRANCH}_md_diff.xlsx \
  --link_res_file ${REPO}_${BRANCH}_md_diff_res \
  --func check_link
