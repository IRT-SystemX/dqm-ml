.EXPORT_ALL_VARIABLES:

# Install dependencies
install:
	python -m pip install --upgrade pip
	pip install -r requirements.txt
	pip install black flake8

	sudo apt-get install python-dev-is-python3 -y
	sudo apt-get install libhunspell-dev -y
	pip install hunspell cspell


# Code quality
format:
	# -----  Formatting Python code with Black
	black .

check:
	# -----  Validating Black code style
	black --check --diff .

	# -----  Validating Flake8 code style
	flake8 .

	# -----  Validating CSpell errors
	cspell .
