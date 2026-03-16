# Frameworks
- Use `uv` for package management.
- Use `pytest` for unit tests.
- Use `typer` to parse command line arguments.
- There will be a script `transcribe` in the `pyproject.toml` file.

# Directory Structure
```
src/
  transcribe/        # main package
    model/           # contains asr models
      model.py      
    cli.py           # command line entry point
tests/              # unit tests
```

# Requirements
- Keep your design decisions in `DESIGN.md`
- Always remember to update `README.md` file to reflect any feature changes that are important to the end user.
- After adding features, remember to update and run unit tests.
