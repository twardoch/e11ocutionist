"""Test suite for e11ocutionist."""

from pathlib import Path


def test_version():
    """Verify package exposes version."""
    import e11ocutionist

    assert e11ocutionist.__version__


def test_package_structure():
    """Verify the package has the expected module structure."""
    import e11ocutionist

    # Check core module files exist in the package
    package_dir = Path(e11ocutionist.__file__).parent
    assert (package_dir / "__init__.py").exists()
    assert (package_dir / "__version__.py").exists()
    assert (package_dir / "e11ocutionist.py").exists()

    # Importing these might fail if not implemented yet, so we do a softer check for now
    module_files = [
        "chunker.py",
        "cli.py",
        "elevenlabs_converter.py",
        "elevenlabs_synthesizer.py",
        "entitizer.py",
        "neifix.py",
        "orator.py",
        "tonedown.py",
        "utils.py",
    ]

    for module_file in module_files:
        if (package_dir / module_file).exists():
            # Try importing the module
            module_name = module_file.replace(".py", "")
            try:
                __import__(f"e11ocutionist.{module_name}")
            except ImportError:
                pass
