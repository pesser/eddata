# Generate baseline images using pytest-mpl


```bash
pip install pytest-mpl
py.test --mpl-generate-path=baseline # generates test images

```

# Run test
```bash
pip install pytest-mpl
py.test --mpl --mpl-baseline-path=baseline # compares against test images in ./baseline
```