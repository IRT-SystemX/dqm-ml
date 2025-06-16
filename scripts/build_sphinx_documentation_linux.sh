# Copy dependencies
cd ..
cp -R _static docs/source/
cp -R examples docs/source/

# Delete old tdaad modules

rm -f docs/source/tdaad*.rst

# Generate package docstring

sphinx-apidoc -o docs/source tdaad

# Generate HTML

cd docs
make clean
make html

rm -Rf source/_static
rm -Rf source/examples
