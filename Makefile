cwd=$(pwd)

cd train
poetry install

cd $cwd

cd train/extraction
poetry install --no-root
poetry export --without-hashes --format=requirements.txt > requirements.txt

cd train/output
poetry install --no-root
poetry export --without-hashes --format=requirements.txt > requirements.txt

cd $cwd
