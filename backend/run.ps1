# Run AGNI backend from backend folder (use quoted path if path has spaces)
Set-Location $PSScriptRoot
python -m uvicorn main:app --reload
