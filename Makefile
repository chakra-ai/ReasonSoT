.PHONY: install test test-integration bench demo lint clean

install:
	pip install -e ".[dev]"

test:
	pytest tests/ -v -m "not integration"

test-integration:
	pytest tests/ -v -m integration

bench:
	python benchmarks/run_benchmarks.py

demo:
	python demo.py

lint:
	python -m py_compile reason_sot/__init__.py
	python -m py_compile config.py

clean:
	find . -type d -name __pycache__ -exec rm -rf {} +
	find . -type f -name "*.pyc" -delete
	rm -rf .pytest_cache benchmarks/results/*.json
