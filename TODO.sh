# bash TODO.sh
find . -type f -name "*.py" | xargs grep -n --color "TODO"
find . -type f -name "*.py" | xargs grep -n --color "XXX"