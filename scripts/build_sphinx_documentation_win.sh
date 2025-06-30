# Copy dependencies
cd ..
cp -R _static docs/source/
cp -R examples docs/source/

# Delete old dqm modules

rm -f docs/source/dqm*.rst

# Generate package docstring

sphinx-apidoc -o docs/source dqm

# Generate HTML

cd docs
./make.bat clean
./make.bat html

# Clean temp directories
rm -Rf source/_static
rm -Rf source/examples
