# Launch pylint
cd ..
mkdir -p _static/pylint
mkdir -p _static/flake8
# Lancement de flake8
flake8 dqm  --config=scripts/.flake8 --format=html --htmldir=_static/flake8 --exit-zero 
SEVERITY=$(xmllint --html --xpath "//*[@id='masthead']" ./_static/flake8/index.html | awk -F 'class="|"/>' '{print $2}')
if [ "$SEVERITY" = "sev-1" ]; then FLAKE8_COLOR="red"
elif [ "$SEVERITY" = "sev-2" ]; then FLAKE8_COLOR="orange"
elif [ "$SEVERITY" = "sev-3" ]; then FLAKE8_COLOR="yellow"
else FLAKE8_COLOR="green"; fi
anybadge --overwrite --label flake8 --value="report" --file=_static/flake8/flake8.svg --color $FLAKE8_COLOR

# Lancement de pylint
pylint dqm --rcfile=scripts/.pylintrc --output-format=text --exit-zero | tee _static/pylint/pylint.txt
PYLINT_SCORE=$(sed -n 's/^Your code has been rated at \([-0-9.]*\)\/.*/\1/p' _static/pylint/pylint.txt)
pylint dqm --rcfile=scripts/.pylintrc --output-format=pylint_gitlab.GitlabPagesHtmlReporter --exit-zero > _static/pylint/index.html 
anybadge --overwrite --label pylint --value=$PYLINT_SCORE --file=_static/pylint/pylint.svg 4=red 6=orange 8=yellow 10=green

# Test coverage

echo "Running test coverage..."
coverage run -m pytest tests
coverage report > scripts/coverage_report.txt
mkdir -p _static/coverage  
COVERAGE_SCORE=$(grep 'TOTAL' scripts/coverage_report.txt | awk '{print $4}' | sed 's/%//')
echo "Coverage score is $COVERAGE_SCORE"
anybadge --overwrite --label coverage --value=$COVERAGE_SCORE --file=_static/coverage/coverage.svg 50=red 60=orange 75=yellow 100=green
coverage html -d _static/coverage
rm -f scripts/coverage_report.txt


