# fix_lint.ps1

Write-Output "Running Ruff autofix..."
ruff check . --fix

Write-Output "Sorting imports with isort..."
isort .

Write-Output "Formatting code with black..."
black .

Write-Output "Running pylint checks..."
pylint .
