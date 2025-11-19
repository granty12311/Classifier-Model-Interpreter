# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Classifier Model Interpreter Package** that provides tools and methodologies for interpreting and explaining machine learning classification models. The package aims to make ML models more transparent and actionable for business stakeholders through various interpretation techniques.

## Architecture

[To be updated as architecture is defined]

### Core Components

[To be defined based on project requirements]

### Data Flow

```
[To be defined]
```

### Module Responsibilities

- **`src/`**: Primary source code (use this for all imports and development)
- **`notebooks/`**: Demo notebooks and analysis examples
- **`tests/`**: Integration and validation tests
- **`data/`**: Sample data and data generation scripts
- **`outputs/`**: Generated charts, reports, and exports

## Development Commands

### Environment Setup

```bash
# Activate virtual environment (use this before any work)
source .venv/bin/activate  # Linux/Mac
# or
.venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

### Running Tests

```bash
# Run all tests
pytest tests/

# Run with coverage
pytest --cov=src tests/
```

### Running Notebooks

```bash
# Start Jupyter
jupyter notebook

# Or execute notebook from command line
jupyter nbconvert --to notebook --execute notebooks/[notebook_name].ipynb
```

### Common Import Pattern

All notebooks and scripts must add `src/` to Python path:

```python
import sys
from pathlib import Path
sys.path.insert(0, str(Path.cwd() / 'src'))

# Now can import modules
# from module_name import function_name
```

## Code Structure Details

[To be defined as modules are created]

## Testing Strategy

**IMPORTANT**: Testing should be done by executing relevant Jupyter notebooks and getting user feedback.

### Testing Workflow

1. **Make code changes** in `src/` directory
2. **Execute relevant notebook** to validate changes
3. **Review outputs** with the user - charts, tables, console output
4. **User provides feedback** on whether results are correct
5. **Iterate** based on feedback

### Which Notebook to Use for Testing

[To be defined as notebooks are created]

### What to Validate

When executing notebooks for testing:
- Code runs without errors
- Outputs match expectations
- Visualizations render correctly
- Results are interpretable and actionable
- User confirms results are valid

## Development Notes

### Performance Considerations

[To be defined based on implementation]

### Best Practices

- ALWAYS prefer editing existing files over creating new ones
- Use clear, descriptive variable and function names
- Document complex logic with comments
- Keep functions focused and modular
- Follow PEP 8 style guidelines

## Git Workflow

### Repository Configuration

- **GitHub URL**: https://github.com/granty12311/Classifier-Model-Interpreter
- **Authentication**: SSH only

### Development Workflow

1. **Make all changes in `src/` directory** during development
2. **Test using notebooks** (see Testing Strategy section)
3. **Get user approval** on changes
4. **Commit and push** to remote

### Git Commands for Pushing

```bash
# Stage all changes
git add .

# Commit with descriptive message
git commit -m "Update: [describe changes here]"

# Push to GitHub using SSH
git push -u origin main
```

### Pre-Push Checklist

Before pushing to GitHub, verify:
- [ ] All changes tested via notebooks with user approval
- [ ] No notebook outputs committed (notebooks should be clean)
- [ ] No sensitive data or large data files included
- [ ] Code follows project conventions and style
- [ ] Dependencies updated in requirements.txt if needed

### Common Git Issues

**Issue**: "Permission denied (publickey)"
- **Solution**: Ensure SSH keys are configured for GitHub: `ssh -T git@github.com`

## Key Files

[To be updated as key files are created]

## Common Development Scenarios

[To be defined based on common use cases]

---

**Note**: This file will be updated as the project evolves. Keep it synchronized with actual project structure and conventions.
