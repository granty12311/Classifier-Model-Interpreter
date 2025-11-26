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
- **IMPORTANT**: Only the `prod_code/` folder is pushed to GitHub. All development happens locally in `src/` and `notebooks/`, then production-ready code is copied to `prod_code/` for deployment.

### Development Workflow

1. **Make all changes in `src/` directory** during development
2. **Test using notebooks** (see Testing Strategy section)
3. **Get user approval** on changes
4. **Deploy to prod_code and push** (see Deployment Workflow below)

### Deployment Workflow (Push to GitHub)

**IMPORTANT**: Only `prod_code/` folder contents are pushed to GitHub. This keeps the public repo clean with only production-ready code.

**Steps:**

1. **Update prod_code folder with latest source code and notebooks**
```bash
cd /home/granty1231/Classifier-Model-Interpreter

# Remove old src to ensure clean copy
rm -rf prod_code/src

# Copy source code (all src/ contents)
cp -r src prod_code/

# Copy demo/template notebooks (all relevant test/demo notebooks)
cp notebooks/demo_all_visualizations.ipynb prod_code/
# Add other template notebooks as needed:
# cp notebooks/other_notebook.ipynb prod_code/

# Copy requirements
cp requirements.txt prod_code/
```

**What to include in prod_code:**
- `src/` - All source code modules
- Template/demo notebooks - Notebooks that demonstrate package usage (e.g., `demo_all_visualizations.ipynb`)
- `requirements.txt` - Package dependencies
- `README.md` - Documentation for users

**What NOT to include:**
- Executed notebooks with outputs (e.g., `*_executed.ipynb`)
- Data files or generated outputs
- Development-only files

2. **Stage and commit prod_code changes**
```bash
# Stage only prod_code folder
git add prod_code/

# Commit with descriptive message (use heredoc for multi-line)
git commit -m "$(cat <<'EOF'
Deploy: [Brief description of changes]

[Detailed description of what changed]

Key updates:
- [List key updates]
- [More updates]

🤖 Generated with [Claude Code](https://claude.com/claude-code)

Co-Authored-By: Claude <noreply@anthropic.com>
EOF
)"
```

3. **Push to GitHub**
```bash
git push origin main
```

### Pre-Push Checklist

Before pushing to GitHub, verify:
- [ ] All changes tested via notebooks with user approval
- [ ] prod_code folder contains: src/, demo_all_visualizations.ipynb, requirements.txt, README.md
- [ ] prod_code/README.md is up-to-date with latest features
- [ ] No sensitive data or large data files included
- [ ] Only prod_code/ is being committed (not local dev files)

### Common Git Issues

**Issue**: "Permission denied (publickey)"
- **Solution**: Ensure SSH keys are configured for GitHub: `ssh -T git@github.com`

## Key Files

[To be updated as key files are created]

## Common Development Scenarios

[To be defined based on common use cases]

---

**Note**: This file will be updated as the project evolves. Keep it synchronized with actual project structure and conventions.
